import numpy as np
import csv
import sys

class MultiClassDataLoader(object):
    """
    Handles multi-class training data.  It takes predefined sets of "train_data_file" and "dev_data_file"
    of the following record format.
        <text>\t<class label>
      ex. "what a masterpiece!	Positive"
    Class labels are given as "class_data_file", which is a list of class labels.
    """
    def __init__(self, flags, data_processor):
        self.__flags = flags
        self.__data_processor = data_processor
        self.__train_data_file = None
        self.__val_data_file = None
        self.__class_data_file = None
        self.__classes_cache = None


    def define_flags(self):
        self.__flags.DEFINE_string("train_data_file", "./data/trainR.csv", "Data source for the training data.")
        self.__flags.DEFINE_string("val_data_file", "./data/valR.csv", "Data source for the cross validation data.")
        self.__flags.DEFINE_string("class_data_file", "./data/reviews.cls", "Data source for the class list.")

    def prepare_data(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_val, y_val = self.__load_data_and_labels(self.__val_data_file)

        # no Attr for str.decode Issue !!!
        # change doc.decode("utf-8") to doc / cause i think them already in utf-8
        # 2018. 11. 19 changes.
        max_doc_len = max([len(doc) for doc in x_train])
        max_doc_len_val = max([len(doc) for doc in x_val])
        if max_doc_len_val > max_doc_len:
            max_doc_len = max_doc_len_val
        # Build vocabulary
        self.vocab_processor = self.__data_processor.vocab_processor(x_train, x_val)
        x_train = np.array(list(self.vocab_processor.fit_transform(x_train)))
        # Build vocabulary
        x_val = np.array(list(self.vocab_processor.fit_transform(x_val)))
        return [x_train, y_train, x_val, y_val]

    def restore_vocab_processor(self, vocab_path):
        return self.__data_processor.restore_vocab_processor(vocab_path)

    def class_labels(self, class_indexes):
        return [ self.__classes()[idx] for idx in class_indexes ]

    def class_count(self):
        return self.__classes().__len__()

    def load_val_data_and_labels(self):
        self.__resolve_params()
        x_val, y_val = self.__load_data_and_labels(self.__val_data_file)
        return [x_val, y_val]

    def load_data_and_labels(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_val, y_val = self.__load_data_and_labels(self.__val_data_file)
        x_all = x_train + x_val
        y_all = np.concatenate([y_train, y_val], 0)
        return [x_all, y_all]

    def __load_data_and_labels(self, data_file):
        x_text = []
        y = []
        with open(data_file, 'r') as tsvin:
            classes = self.__classes()
            # 멀티핫으로 대체할 것. 배열화시키자 0100001 같이 !
            one_hot_vectors = np.eye(len(classes), dtype=int)
            class_vectors = {}
            for i, cls in enumerate(classes):
                class_vectors[cls] = one_hot_vectors[i]

            print("##### Mapping Classes Completed : ", data_file)

            # 클래스 벡터 생성 후에 각 라인에 대해 리뷰 데이터 마다 라벨 벡터를 연결할 것.
            # 형태소 분석 정제 처리
            tsvin = csv.reader(tsvin, delimiter=',')
            next(tsvin, None)

            for row in tsvin:
                data = self.__data_processor.clean_data(row[0])
                x_text.append(data)
                vector = np.zeros(len(classes), dtype=int)

                if row[1] is not '':
                    for r in row[1].split('/'):
                        vector = vector + class_vectors[r]
                y.append(vector)
        return [x_text, np.array(y)]

    def __classes(self):
        self.__resolve_params()
        if self.__classes_cache is None:
            with open(self.__class_data_file, 'r') as list_class:
                classes = list(list_class.readlines())
                self.__classes_cache = [s.strip() for s in classes]
        return self.__classes_cache

    def __resolve_params(self):
        if self.__class_data_file is None:
            self.__train_data_file = self.__flags.FLAGS.train_data_file
            self.__val_data_file = self.__flags.FLAGS.val_data_file
            self.__class_data_file = self.__flags.FLAGS.class_data_file

    def printProgress (self, iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
        formatStr = "{0:." + str(decimals) + "f}"
        percent = formatStr.format(100 * (iteration / float(total)))
        filledLength = int(round(barLength * iteration / float(total)))
        bar = '#' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()