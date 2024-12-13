from Bio import SeqIO

a_pro = set()
a_seq = set()
file = open('./independent_set_test.fasta').readlines()
num1 = 0
for xulie in file:
    num1 += 1
    if num1 % 2 == 1:
        pro = xulie.strip()[8:]
        # print(pro)
        a_pro.add(pro)
    else:
        seq1 = xulie.strip()
        a_seq.add(seq1)

b_pro = set()
b_seq = set()
file2 = open('./fold_train.fasta').readlines()
num2 = 0
for xulie2 in file2:
    num2 += 1
    if num2 % 2 == 1:
        pro2 = xulie2.strip()[8:]
        b_pro.add(pro2)
    else:
        seq2 = xulie2.strip()
        b_seq.add(seq2)

if len(a_pro & b_pro) == 0 and len(a_seq & b_seq) == 0:
    print("The protein types in the independent set do not overlap with those in the training set, and their sequences are unique.")
else:
    print("The protein types in the independent set overlap with those in the training set, or their sequences are not unique.")



fold1 = set()
fold2 = set()
fold3 = set()
fold4 = set()
fold5 = set()


for f1 in SeqIO.parse('./train_fold1_1.fasta', 'fasta'):
    fold1.add(f1.name[7:])
for f2 in SeqIO.parse('./train_fold1_2.fasta', 'fasta'):
    fold1.add(f2.name[7:])
for f3 in SeqIO.parse('./train_fold1_3.fasta', 'fasta'):
    fold1.add(f3.name[7:])
for f4 in SeqIO.parse('./train_fold1_4.fasta', 'fasta'):
    fold1.add(f4.name[7:])
for f5 in SeqIO.parse('./train_fold1_5.fasta', 'fasta'):
    fold1.add(f5.name[7:])

sets = [fold1, fold2, fold3, fold4, fold5]
# # result = all(len(set(sets[i]).intersection(sets[j])) == 0 for i in range(len(sets)) for j in range(i + 1, len(sets)))
result = True
#

for i in range(len(sets)):
    for j in range(i + 1, len(sets)):

        if len(sets[i].intersection(sets[j])) > 0:
            result = False
            break
    if not result:
        break

if result:
    print("The protein types in each fold dataset are mutually independent.")
else:
    print("The 5-fold datasets contain overlapping protein types.")


