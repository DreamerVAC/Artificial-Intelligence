import numpy as np
def BinarySearch(nums, target):
    """
    :param nums: list[int]
    :param target: int
    :return: int
    """
    if len(nums) == 0:
        return -1
    mid = len(nums) // 2
    if nums[mid] == target:
        return mid
    if nums[mid] > target:
        return BinarySearch(nums[:mid], target)
    else:
        return BinarySearch(nums[mid + 1:], target)


def MatrixAdd(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    n = len(A[0])
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C


def MatrixMul(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    n = len(A[0])
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


def ReverseKeyValue(dict1):
    """
    :param dict1: dict
    :return: dict
    """
    dict2 = {}
    for key, value in dict1.items():
        dict2[value] = key
    return dict2


def main():
    A = [1, 3, 4, 2, 5, 6, 8, 7]
    target = int(input("请输入目标值: "))
    n1 = BinarySearch(A, target)
    print("The index of target is ", n1)
    B = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    C = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
    D = MatrixAdd(B, C)
    E = MatrixMul(B, C)
    print("The added matrix is ", D)
    print("The multiplication matrix is ", E)
    F = {'cat': 'cute', 'dog': 'big', 'horse': 'handsome'}
    G = ReverseKeyValue(F)
    print('The reversed dictionary is ', G)


if __name__ == '__main__':
    main()
