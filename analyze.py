
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)


#загрузка и поготовка
df = pd.read_csv('students.csv', encoding='utf-8-sig')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(['ProcessInstanceId', 'EventIndex']).reset_index(drop=True)

print('ЗАГРУЗКА')
print(f'Событий: {len(df)}')
print(f'Процессов: {df["ProcessInstanceId"].nunique()}')
print('\nУникальные ActivityName:')
for a in sorted(df['ActivityName'].dropna().unique()):
    print('  -', repr(a))


# Сводка процессов
proc = df.groupby('ProcessInstanceId').agg(
    RequestType=('RequestType', 'first'),
    Amount=('Amount', 'first'),
    Result=('Result', 'last'),
    Start=('Timestamp', 'min'),
    End=('Timestamp', 'max'),
    Events=('EventId', 'count'),
).reset_index()
proc['Duration_h'] = (proc['End'] - proc['Start']).dt.total_seconds() / 3600

seq = (
    df.sort_values(['ProcessInstanceId', 'EventIndex'])
      .groupby('ProcessInstanceId')['ActivityName']
      .apply(lambda x: ' -> '.join(x.tolist()))
      .reset_index(name='Sequence')
)
proc = proc.merge(seq, on='ProcessInstanceId', how='left')

print('\nСтатусы процессов:')
print(proc['Result'].value_counts(dropna=False).to_string())


# Возвраты
return_pids = set(df[df['ActivityName'] == 'Доработка заявки']['ProcessInstanceId'])
return_pids |= set(
    df[(df['ActivityName'] == 'Согласование руководителем') &
       (df['Result'] == 'return')]['ProcessInstanceId']
)
rejected_pids = set(
    df[(df['ActivityName'] == 'Завершение') &
       (df['Result'] == 'rejected')]['ProcessInstanceId']
)
print(f'\nПроцессов с возвратом: {len(return_pids)}')
print(f'Отклонённых процессов: {len(rejected_pids)}')


# Справки
mandatory_standard = [
    'Регистрация заявки', 'Проверка заявки', 'Согласование руководителем',
    'Согласование бюджета', 'Исполнение заявки', 'Завершение',
]
mandatory_contract = [
    'Регистрация заявки', 'Проверка заявки', 'Согласование руководителем',
    'Согласование бюджета', 'Проверка договора', 'Исполнение заявки', 'Завершение',
]
role_map = {
    'Проверка заявки': 'specialist',
    'Согласование руководителем': 'manager',
    'Согласование бюджета': 'finance',
    'Проверка договора': 'legal',
    'Исполнение заявки': 'procurement',
    'Регистрация заявки': 'requester',
}
duration_limits_h = {
    'Проверка заявки': 24,
    'Согласование руководителем': 72,
    'Согласование бюджета': 72,
    'Проверка договора': 120,
    'Исполнение заявки': 168,
    'Доработка заявки': 72,
}


# Поиск аномалий
anomalies = []

for _, row in proc.iterrows():
    pid = row['ProcessInstanceId']
    req = row['RequestType']
    res = row['Result']

    g = df[df['ProcessInstanceId'] == pid]
    acts_set = set(g['ActivityName'].tolist())
    was_return = pid in return_pids

    # 5.1. Отсутствует обязательный этап
    if res == 'fulfilled' and not was_return:
        mand = mandatory_contract if req == 'contract' else mandatory_standard
        for m in mand:
            if m not in acts_set:
                anomalies.append((
                    pid, 'Отсутствует обязательный этап',
                    f'Успешная заявка типа {req} не содержит этап "{m}"'
                ))

    # 5.2. Contract без проверки договора
    if res == 'fulfilled' and req == 'contract' and 'Проверка договора' not in acts_set:
        anomalies.append((
            pid, 'Нет проверки договора',
            'Договорная заявка завершена без юридической проверки'
        ))

    # 5.3. Несоответствие роли
    for _, ev in g.iterrows():
        act, role = ev['ActivityName'], ev['Role']
        if act in role_map and pd.notna(role) and role not in ('system', 'auditor'):
            if role != role_map[act]:
                anomalies.append((
                    pid, 'Несоответствие роли',
                    f'Этап "{act}" выполнен ролью {role}, ожидается {role_map[act]}'
                ))

    # 5.4. Нарушен порядок времени
    times = g.sort_values('EventIndex')['Timestamp'].tolist()
    if any(times[i] > times[i+1] for i in range(len(times)-1)):
        anomalies.append((
            pid, 'Нарушен порядок времени',
            'Метки Timestamp не возрастают по EventIndex'
        ))

    # 5.5. События после Завершения
    comp = g[g['ActivityName'] == 'Завершение']
    if not comp.empty:
        end_idx = comp['EventIndex'].min()
        after = g[g['EventIndex'] > end_idx]
        if not after.empty:
            anomalies.append((
                pid, 'События после завершения',
                f'После "Завершение" зафиксировано ещё {len(after)} событий'
            ))


# Длительность аномалии
tasks = df[df['TaskId'].notna()]
rows = []
for tid, tg in tasks.groupby('TaskId'):
    created = tg[tg['EventType'] == 'task_created']['Timestamp']
    completed = tg[tg['EventType'] == 'task_completed']['Timestamp']
    if not created.empty and not completed.empty:
        dur = (completed.min() - created.min()).total_seconds() / 60
        rows.append({
            'ProcessInstanceId': tg['ProcessInstanceId'].iloc[0],
            'ActivityName': tg['ActivityName'].iloc[0],
            'RequestType': tg['RequestType'].iloc[0],
            'Role': tg['Role'].iloc[0] if pd.notna(tg['Role'].iloc[0]) else None,
            'Duration_min': dur,
        })
task_dur = pd.DataFrame(rows)

for _, t in task_dur.iterrows():
    limit_h = duration_limits_h.get(t['ActivityName'])
    if limit_h and t['Duration_min'] > limit_h * 60:
        anomalies.append((
            t['ProcessInstanceId'], 'Аномальная длительность',
            f'{t["ActivityName"]} ({t["RequestType"]}) длилась '
            f'{t["Duration_min"]/60:.1f} ч, порог {limit_h} ч'
        ))


# Сохранение
anom_df = pd.DataFrame(
    anomalies, columns=['ProcessInstanceId', 'AnomalyType', 'Explanation']
).drop_duplicates().reset_index(drop=True)

anom_df.to_csv('anomalies.csv', index=False, encoding='utf-8-sig')
task_dur.to_csv('task_durations.csv', index=False, encoding='utf-8-sig')


# Итог
print('ИТОГИ')
print(f'Всего процессов: {len(proc)}')
print(f'  fulfilled: {(proc["Result"] == "fulfilled").sum()}')
print(f'  rejected:  {(proc["Result"] == "rejected").sum()}')
print(f'  открытых:  {(~proc["Result"].isin(["fulfilled","rejected"])).sum()}')
print(f'\nНайдено аномалий: {len(anom_df)}')
print('\nРаспределение по типам:')
print(anom_df['AnomalyType'].value_counts().to_string())

print('\nПервые 20 строк anomalies.csv:')
print(anom_df.head(20).to_string(index=False))
