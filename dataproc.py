import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re
from pprint import pprint
from pandas.tseries.offsets import DateOffset
from functools import reduce
from itertools import chain

rname = "useratings_"
usnames = [f"{rname}p{c}.json" for c in range(1, 4)]

alldata = pd.concat([pd.read_json(Path(usnames[x]))
                     for x in range(len(usnames))], ignore_index=True)

p = re.compile(r"^\d{1,2}$")
data = alldata.loc[alldata["username"].astype(str).str.match(p)]

data.reset_index(drop=True, inplace=True)

mapping = {"1": "Thanh",
           "2": "Ghafar",
           "3": "Saeed",
           "4": "Carol",
           "5": "Peter",
           "6": "Kitty",
           "7": "Harry",
           "8": "Michael",
           "9": "Melvine",
           "10": "Ahmed",
           "11": "Godfrey",
           "12": "Idriss",
           "13": "Phillippe",
           "14": "Sayima",
           "15": "Emmanuel",
           "16": "Edith",
           "17": "Francis",
           "18": "Frederika",
           "19": "Suhbaa",
           "20": "Mansi",
           "21": "Aditya",
           "22": "Eberechukwu",
           "23": "Mir Ikramul",
           "24": "Janine James",
           "25": "Miriam",
           "26": "Rozenn"}

data = data.assign(username=data.username.astype(str).convert_dtypes(),
                   task0=data.task0.astype(str).convert_dtypes(),
                   task1=data.task1.astype(str).convert_dtypes(),
                   task2=data.task2.astype(str).convert_dtypes(),
                   task3=data.task3.astype(str).convert_dtypes(),
                   password=data.password.astype(str).convert_dtypes())
data = data.assign(fn=data.username.map(mapping))

cols = ['_ts', 'fn', 'username', 'task0', 'task1', 'task2', 'task3', 'password', 'Dpc', 'task0NumSubmissions',
        'task1NumSubmissions', 'task2NumSubmissions', 'task3NumSubmissions',
        'task0Rating', 'task1Rating', 'task2Rating', 'task3Rating']
data = data[cols].reset_index(drop=True).copy(deep=True)

data = data.groupby(["fn"]).filter(lambda x: x['_ts'].count() >= 8)
data["index"] = data.index

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

indices = []
for name in mapping.values():
    tr = data.loc[data["fn"] == name]
    sorted_tr = tr.sort_values(by="_ts")
    if sorted_tr.shape[0] < 8: continue
    sorted_tr["reduced_ts"] = sorted_tr["_ts"] - sorted_tr["_ts"].tolist()[0]
    sorted_tr["days"] = sorted_tr["reduced_ts"] / (24 * 60 * 60)
    # print(sorted_tr)

    start = 0.0
    count = 0
    for x in range(sorted_tr.shape[0]):
        if count >= 7: continue
        if x == 0: indices.append(sorted_tr["index"].tolist()[0])
        else:
            if start + 0.7 < sorted_tr["days"].tolist()[x]:
                indices.append(sorted_tr["index"].tolist()[x])
                start = sorted_tr["days"].tolist()[x]
                count += 1

data = data.loc[indices]
data = data.groupby(["fn"]).filter(lambda x: x['_ts'].count() >= 8)

completed_status = data.groupby('fn').count()['_ts']
completed_so_far = len(data.groupby("fn").count()["_ts"])

print("Completed so far: \n---------------\n", completed_status,
      "\n---------------\nTotal people: ", completed_so_far)

n = 8
df = data.groupby("fn").apply(lambda x: x.sample(n=n, random_state=42))

# exit()
print(df)
# save_file = df.drop(columns=["fn"])
# print(save_file)
# df.to_csv("filtered_data.csv")
# exit()
ttrating = df.loc[:, cols[0:7]+cols[9:]]

# print(ttrating)
# exit()


def task_to_rating_mapping(row): return [(
    row[f'task{x}'], row[f'task{x}Rating'], row["_ts"], row["username"]) for x in range(4)]


def task_to_submission_mapping(row): return [(
    row[f'task{x}'], row[f'task{x}NumSubmissions'], row["_ts"], row["username"]) for x in range(4)]


tasktypes = ["linux", "normal", "personalized", "standard"]


def get_task_to_key(subdf, mapping_function):
    tuple_series = subdf.apply(mapping_function, axis=1).values
    return list(chain.from_iterable(tuple_series))


# def get_cols_of_X_given_Y(X, Y, row_by_row, groupbyKeys):
#     row_by_row_grouped = row_by_row.groupby(X).agg(
#         lambda groupng: list(groupng)).unstack()[Y]
#     nt2row_by_row_dict = dict(
#         zip(groupbyKeys, [row_by_row_grouped[k] for k in groupbyKeys]))
#     return pd.DataFrame(nt2row_by_row_dict)


t2rdf = pd.DataFrame(data=get_task_to_key(
    ttrating, task_to_rating_mapping), columns=["Task", "Rating", "_ts", "username"])
# t2rdf = pd.DataFrame(data=get_task_to_key(
#     ttrating, task_to_submission_mapping), columns=["Task", "Rating", "_ts", "username"])

print(ttrating)
print(t2rdf)

# nt2rdf = get_cols_of_X_given_Y(
#     X="Task", Y="Rating", row_by_row=t2rdf, groupbyKeys=tasktypes)

# fig, ax = plt.subplots(figsize=(18, 12))
# nt2rdf.plot.box(ax=ax, fontsize="xx-large")
# ax.set_title("Ratings for Task Types", fontsize="xx-large")
# plt.savefig("ratings2tasktypes.png")
# ax.plot()
# plt.show()

# ttsubs = df.loc[:, cols[3:7]+cols[9:13]]


# def task_to_subs_mapping(row): return [(
#     row[f'task{x}'], row[f'task{x}NumSubmissions']) for x in range(4)]


# t2sdf = pd.DataFrame(data=get_task_to_key(
#     ttsubs, task_to_subs_mapping), columns=["Task", "NumSubmissions"])
# t2sdf = t2sdf.assign(NoMistake=lambda x: x.NumSubmissions == 2)
# nt2sdf = get_cols_of_X_given_Y(
#     X="Task", Y="NoMistake", row_by_row=t2sdf, groupbyKeys=tasktypes)

# nt3sdf = nt2sdf.apply(lambda cat: np.unique(cat, return_counts=True)) \
#     .apply(lambda x: dict(zip(*x))).to_frame() \
#     .transpose() \
#     .applymap(lambda x: round((x[False]/(x[False]+x[True]))*100, 2))

# print(nt3sdf)

# fntsize = 'xx-large'
# fig, ax = plt.subplots(figsize=(18, 12))
# nt3sdf.plot.bar(ax=ax, fontsize=fntsize)
# ax.set_title("Percentage of mistakes per type", fontsize=fntsize)
# ax.xaxis.set_ticks([])
# ax.xaxis.set_label_text("Task types", fontsize=fntsize)
# ax.yaxis.set_label_text("Percentage of mistakes", fontsize=fntsize)
# ax.legend(fontsize=fntsize)
# plt.savefig("Percentage of mistakes per type.png")
# ax.plot()
# plt.show()

# df.groupby("password").agg({'Dpc': list}).unstack().to_frame(
# ).transpose()  # .count() # Need to compute anova for this

# subs_by_password = df.groupby("password").agg(
#     {f"task{x}NumSubmissions": list for x in range(4)})

# subs_by_password = subs_by_password.assign(
#     allsubmissions=subs_by_password.task0NumSubmissions
#     + subs_by_password.task1NumSubmissions
#     + subs_by_password.task2NumSubmissions
#     + subs_by_password.task3NumSubmissions
# )[["allsubmissions"]]

# nsbp = subs_by_password.transpose().applymap(
#     lambda x: [v == 2 for v in x]).applymap(lambda x: np.unique(x, return_counts=True))

# nsbp = nsbp.applymap(lambda x: dict(
#     zip(*x))).applymap(lambda x: round((x[False]/(x[False]+x[True]))*100, 2))

# print(nsbp)

# fntsize = 'xx-large'
# fig, ax = plt.subplots(figsize=(18, 12))
# nsbp.plot.box(ax=ax, fontsize=fntsize)
# ax.set_title("Percentage of mistakes per passwords", fontsize=fntsize)
# ax.xaxis.set_ticks([])
# ax.xaxis.set_label_text("Passwords", fontsize=fntsize)
# ax.yaxis.set_label_text("Percentage of mistakes", fontsize=fntsize)
# ax.legend(fontsize=fntsize)
# plt.savefig("Percentage of mistakes per password.png")
# ax.plot()
# plt.show()

# rtingsbypasswords = df.groupby("password").agg(
#     {f"task{x}Rating": list for x in range(4)})

# rtingsbypasswords = rtingsbypasswords.assign(
#     allratings=rtingsbypasswords.task0Rating
#     + rtingsbypasswords.task1Rating
#     + rtingsbypasswords.task2Rating
#     + rtingsbypasswords.task3Rating
# )[["allratings"]]

# rtingsbypasswords = rtingsbypasswords.transpose()
# rtingsbypasswords.apply(lambda x: pd.Series(x.values[0])).describe()
# rtingsbypasswords.apply(lambda x: pd.Series(x.values[0])).describe().to_html()
# rtbp2 = rtingsbypasswords.apply(lambda x: pd.Series(x.values[0]))

# fntsize = 'xx-large'
# fig, ax = plt.subplots(figsize=(18, 12))
# rtbp2.plot.box(ax=ax, fontsize=fntsize)
# ax.set_title("Ratings for passwords", fontsize=fntsize)
# ax.xaxis.set_ticks([])
# ax.xaxis.set_label_text("Passwords", fontsize=fntsize)
# ax.yaxis.set_label_text("Ratings", fontsize=fntsize)
# # ax.legend(fontsize=fntsize)
# # plt.savefig("Percentage of mistakes per password.png")
# ax.plot()
# plt.show()

# nmapping = {f'{i}': key for i, key in enumerate(
#     [chr(x+64) for x in range(27)])}
# username_to_rounds = df.assign(username=df.username.apply(lambda x: f"Person {nmapping[x]}")).groupby(
#     "username").agg(dict(zip([f"task{x}Rating" for x in range(4)], [list for _ in range(4)])))

# u2r = username_to_rounds.reset_index(drop=True)
# u2r = u2r.assign(username=list(map(lambda x: f"Person {nmapping[str(x+1)]}", u2r.index.values))).set_index("username")

# print(u2r)

def find_change_over_days(data, column):
    res = dict()
    counter = 0

    total = []
    for x in range(data.shape[0]):
        row = data.loc[x]
        if (row[1] != column): continue

        data_row = row[2]
        for y in range(len(data_row)):
            total.append(data_row[y])
            if y in res:
                res[y] += data_row[y]
            else:
                res[y] = data_row[y]
        counter += 1

    resvec = []
    for key in res.keys():
        res[key] = res[key] / counter
        resvec.append(res[key])

    print(resvec)
    # base = len(total) * 2.0
    # print(base)
    # print(sum(total))
    # print((sum(total) - base) / sum(total))
    print(np.mean(resvec))
    print(np.std(total))
    return resvec

def grab_values_at_index(data, column, index):
    res = dict()
    counter = 0

    total = []
    for x in range(data.shape[0]):
        row = data.loc[x]
        if (row[1] != column): continue

        data_row = row[2]
        total.append(data_row[index])

    return total

# task0 = find_change_over_days(u2r, "task0Rating")
# task1 = find_change_over_days(u2r, "task1Rating")
# task2 = find_change_over_days(u2r, "task2Rating")
# task3 = find_change_over_days(u2r, "task3Rating")

print(df)
sorted_df = t2rdf.sort_values(by='_ts')
print(sorted_df)

nmapping = {f'{i}': key for i, key in enumerate(
    [chr(x+64) for x in range(27)])}
username_to_rounds = sorted_df.groupby(
    ["username", "Task"]).agg(dict(zip([f"Rating" for x in range(4)], [list for _ in range(4)])))

username_to_rounds = username_to_rounds.reset_index()
print(username_to_rounds)
print(username_to_rounds.loc[1])

# u2r = username_to_rounds.reset_index(drop=True)
# u2r = u2r.assign(username=list(map(lambda x: f"Person {nmapping[str(x+1)]}", u2r.index.values))).set_index("username")
# print(u2r)

task0 = find_change_over_days(username_to_rounds, "linux")
task1 = find_change_over_days(username_to_rounds, "normal")
task2 = find_change_over_days(username_to_rounds, "personalized")
task3 = find_change_over_days(username_to_rounds, "standard")

linux_0 = grab_values_at_index(username_to_rounds, "linux", 0)
normal_0 = grab_values_at_index(username_to_rounds, "normal", 0)
pers_0 = grab_values_at_index(username_to_rounds, "personalized", 0)
stand_0 = grab_values_at_index(username_to_rounds, "standard", 0)

linux_7 = grab_values_at_index(username_to_rounds, "linux", 7)
normal_7 = grab_values_at_index(username_to_rounds, "normal", 7)
pers_7 = grab_values_at_index(username_to_rounds, "personalized", 7)
stand_7 = grab_values_at_index(username_to_rounds, "standard", 7)

from scipy.stats import wilcoxon

res_linux_normal = wilcoxon(linux_0, normal_0, alternative="two-sided")
print(res_linux_normal.statistic)
print(res_linux_normal.pvalue)

res_linux_pers = wilcoxon(linux_0, pers_0, alternative="two-sided")
print(res_linux_pers.statistic)
print(res_linux_pers.pvalue)

res_linux_standard = wilcoxon(linux_0, stand_0, alternative="two-sided")
print(res_linux_standard.statistic)
print(res_linux_standard.pvalue)


res_linux_normal7 = wilcoxon(linux_7, normal_7, alternative="two-sided")
print(res_linux_normal7.statistic)
print(res_linux_normal7.pvalue)

res_linux_pers7 = wilcoxon(linux_7, pers_7, alternative="two-sided")
print(res_linux_pers7.statistic)
print(res_linux_pers7.pvalue)

res_linux_standard7 = wilcoxon(linux_7, stand_7, alternative="two-sided")
print(res_linux_standard7.statistic)
print(res_linux_standard7.pvalue)

res_linux0_linux7 = wilcoxon(linux_0, linux_7, alternative="two-sided")
print(res_linux0_linux7.statistic)
print(res_linux0_linux7.pvalue)

res_pers0_pers7 = wilcoxon(pers_0, pers_7, alternative="two-sided")
print(res_pers0_pers7.statistic)
print(res_pers0_pers7.pvalue)

# res_linux_normal = wilcoxon(task0, task1, alternative="two-sided")
# print(res_linux_normal.statistic)
# print(res_linux_normal.pvalue)

# res_linux_pers = wilcoxon(task0, task2, alternative="two-sided")
# print(res_linux_pers.statistic)
# print(res_linux_pers.pvalue)

# res_linux_standard = wilcoxon(task0, task3, alternative="two-sided")
# print(res_linux_standard.statistic)
# print(res_linux_standard.pvalue)

# res_normal_pers = wilcoxon(task1, task2, alternative="two-sided")
# print(res_normal_pers.statistic)
# print(res_normal_pers.pvalue)

# res_normal_standard = wilcoxon(task1, task3, alternative="two-sided")
# print(res_normal_standard.statistic)
# print(res_normal_standard.pvalue)

# res_personalized_standard = wilcoxon(task2, task3, alternative="two-sided")
# print(res_personalized_standard.statistic)
# print(res_personalized_standard.pvalue)


def calculate_wilcoxon_signed_rank_test(series_one, series_two):
    abs_list = []
    sign_list = []

    for idx in range(len(series_one)):
        base = series_two[idx] - series_one[idx]
        if abs(base) == 0: continue

        abs_list.append(abs(base))
        sign = 0
        if base > 0:
            sign = 1
        elif base < 0:
            sign = -1

        abs_list.append(sign)
    
    Nr = len(abs_list)

