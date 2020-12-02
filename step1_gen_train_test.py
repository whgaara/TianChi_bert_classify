import random

from pretrain_config import SourcePath, CorpusPath, TDemoPath, EvalPath, EDemoPath, Assistant
from pretrain_config import get_time


class PretrainProcess(object):
    def __init__(self):
        self.src_lines = open(SourcePath, 'r', encoding='utf-8').readlines()
        self.f_train = open(CorpusPath, 'w', encoding='utf-8')
        self.f_tdemo = open(TDemoPath, 'w', encoding='utf-8')
        self.f_eval = open(EvalPath, 'w', encoding='utf-8')
        self.f_edemo = open(EDemoPath, 'w', encoding='utf-8')

        self.source_data_set = {}
        self.train_data_set = {}
        self.eval_data_set = {}
        self.class2count = {}

        self.max_char_num = 0
        self.max_label_num = 0
        self.max_sentence_length = 0

    def traverse(self):
        f = open(Assistant, 'w', encoding='utf-8')
        for i, line in enumerate(self.src_lines):
            if i == 0:
                continue
            if line:
                label, desc = line.split('\t')
                label = int(label)
                desc_list = desc.split(' ')
                desc_list = [int(x) for x in desc_list]

                if self.max_char_num < max(desc_list):
                    self.max_char_num = max(desc_list)
                if self.max_label_num < label:
                    self.max_label_num = label
                if self.max_sentence_length < len(desc_list):
                    self.max_sentence_length = len(desc_list)

                if label in self.source_data_set:
                    self.source_data_set[label].append(desc_list)
                else:
                    self.source_data_set[label] = []
                    self.source_data_set[label].append(desc_list)

        for label, items in self.source_data_set.items():
            self.class2count[label] = len(items)
            random.shuffle(self.source_data_set[label])

        # 补充了cls和padding两个字符
        f.write(str(self.max_char_num + 2) + ',' +
                str(self.max_label_num + 1) + ',' +
                str(self.max_sentence_length) + '\n')
        f.close()

    def gen_num_sub_list(self, source_list, tar_num, ordinal=True):
        assert source_list and tar_num
        if not ordinal:
            source_list.reverse()
        i = 0
        target_list = []
        length = len(source_list)
        while True:
            if len(target_list) >= tar_num:
                break
            if i < length:
                target_list.append(source_list[i])
                i += 1
            else:
                i = 0
        return target_list

    def balance(self):
        train_base_num = sorted(self.class2count.values())[1]
        eval_base_num = train_base_num // 10
        for label, items in self.source_data_set.items():
            if label not in self.train_data_set:
                self.train_data_set[label] = []
            if label not in self.eval_data_set:
                self.eval_data_set[label] = []
            self.train_data_set[label] = self.gen_num_sub_list(items, train_base_num)
            self.eval_data_set[label] = self.gen_num_sub_list(items, max(eval_base_num, 1), False)

    def dump_train_eval(self):
        for label, items in self.train_data_set.items():
            for item in items:
                item = [str(x) for x in item]
                self.f_train.write(str(label) + ',' + ' '.join(item) + '\n')
                if random.random() < 0.1:
                    self.f_tdemo.write(str(label) + ',' + ' '.join(item) + '\n')
        for label, items in self.eval_data_set.items():
            for item in items:
                item = [str(x) for x in item]
                self.f_eval.write(str(label) + ',' + ' '.join(item) + '\n')
                if random.random() < 0.01:
                    self.f_edemo.write(str(label) + ',' + ' '.join(item) + '\n')

    def terminate(self):
        self.f_train.close()
        self.f_eval.close()


if __name__ == '__main__':
    pp = PretrainProcess()
    print(get_time())
    pp.traverse()
    print(get_time())
    pp.balance()
    print(get_time())
    pp.dump_train_eval()
    print(get_time())
    pp.terminate()
