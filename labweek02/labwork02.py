class StuData:
    def __init__(self, infile="student_data.txt"):
        self.infile = infile
        self.data = []
        with open(infile, 'r') as file:
            for line in file:
                stu_info = line.strip().split()
                self.data.append(stu_info)

    def AddData(self, name, stu_num, gender, age):
        self.data.append([name, stu_num, gender, age])

    def SortData(self, attribute):
        index = {'name': 0, 'stu_num': 1, 'gender': 2, 'age': 3}
        self.data.sort(key=lambda x: x[index[attribute]])

    def ExportFile(self, outfile='new_stu_data.txt'):
        with open(outfile, 'w') as file:
            for stu_info in self.data:
                file.write(' '.join(map(str, stu_info)) + '\n')


def main():
    s = StuData()
    s.AddData('Harry', '001', 'M', 10)
    s.SortData('stu_num')
    s.ExportFile()


if __name__ == '__main__':
    main()
