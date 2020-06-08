from main import args
from collections import Counter
import matplotlib.pyplot as plt
import json

print(args.log_dir)

with open(args.init_data, 'r', encoding='utf-8') as f:
    init_data = json.load(f)

init_labels = [i_data['intent'] for i_data in init_data]
init_counter = Counter(init_labels)
#print(init_counter.most_common())
fig, ax = plt.subplots()
x, y = zip(*init_counter.most_common())
x = [i for i in range(23)]
# 截尾平均数
means = sum(sorted(y)[1:-1])/len(y[1:-1])
b = ax.bar(x, y, label='{}'.format(means))
plt.title('Recommended song list score')
for a, b in zip(x, y):
    ax.text(a, b+1, b, ha='center', va='bottom')

plt.xlim((1,23))
plt.ylim((1,y[0]))
plt.xticks(range(len(x)+2))
plt.xlabel('playlist number')
plt.ylabel('score')
plt.legend()
plt.show()
plt.savefig('./数据分布直方图.png', format='png')
