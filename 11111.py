import codecs
import sys

def lang8_right_maker(file_in, file_out):
    with codecs.open(file_in, 'r') as f1:
        with codecs.open(file_out, 'w') as f2:
            for line in f1.readlines():
                wrong, right = line.strip().split('\t')
                f2.write(right + '\n')


# with codecs.open('./bbc_test.txt', 'r') as f1:
#     with codecs.open('./bbc_test_wrong.txt', 'r') as f2:
#         with codecs.open('./bbc_wrong.txt', 'w') as f3:
#             rights = f1.readlines()
#             wrongs = f2.readlines()
#             l = len(rights)
#             for i in range(l):
#                 f3.write(wrongs[i].strip() + '\t' + rights[i].strip() + '\n')



if __name__ == '__main__':
    args = sys.argv
    file_in = args[1]
    file_out = args[2]
    lang8_right_maker(file_in, file_out)