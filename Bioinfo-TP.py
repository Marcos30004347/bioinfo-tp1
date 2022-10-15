import numpy as np
import sys

cost = -1

BLOSUM62 = [
 [4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],
 [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],
 [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],
 [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],
 [0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
 [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],
 [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
 [0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],
 [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],
 [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],
 [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],
 [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],
 [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],
 [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],
 [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
 [1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4], 
 [0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],
 [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],
 [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],
 [0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],
 [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],
 [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
 [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],
 [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1]
]


index = {
    'A':0,  
    'R':1,  
    'N':2,  
    'D':3, 
    'C':4,  
    'Q':5,  
    'E':6,  
    'G':7,  
    'H':8,
    'I':9,
    'L':10,  
    'K':11,  
    'M':12,  
    'F':13,  
    'P':14,
    'S':15,
    'T':16,  
    'W':17,  
    'Y':18,  
    'V':19,
    'B':20,
    'Z':21,
    'X':22,
    '*':23
}

DIAG = 0
UP = 1
LEFT = 2

GAP_SCORE = -4

def needleman_wunsch(v, w):            
    size_v = len(v)
    size_w = len(w)

    s = np.zeros((size_v + 1, size_w + 1), dtype=np.float64)
    b = np.zeros((size_v + 1, size_w + 1), dtype=np.int32)
    
    for i in range(0, size_v + 1):
        b[i, 0] = LEFT
    for i in range(0, size_w + 1):
        b[0, i] = UP
    
    for i in range(1, size_v + 1):
        for j in range(1, size_w + 1):
            diag_score = s[i - 1, j - 1] + BLOSUM62[index[v[i - 1]]][index[w[j - 1]]]
            up_score = s[i, j - 1] + GAP_SCORE
            left_score = s[i - 1, j] + GAP_SCORE

            s[i, j] = max(left_score, up_score, diag_score)            
            
            if s[i, j] == diag_score:
                b[i, j] = DIAG
            elif s[i, j] == up_score:
                b[i, j] = UP
            else:
                b[i, j] = LEFT

    # print(s)
    return s[s.shape[0] - 1, s.shape[1] - 1], b


def printReadableMatrix(path_matrix):
    for i in range(path_matrix.shape[0]):
        for j in range(path_matrix.shape[1]):
            if path_matrix[i][j] == 1:
                print("U ", end=" ")
            elif path_matrix[i][j] == 0:
                print("D ", end=" ")
            elif path_matrix[i][j] == 2:
                print("L ", end=" ")
        print('')


def get_alignment(path_matrix, seq1, seq2):
        align1 = ""
        matches = ""
        align2 = ""

        i = len(seq1) - 1
        j = len(seq2) - 1

        while i != -1 or j != -1:
            path = path_matrix[i + 1][j + 1]

            species1 = seq1[i]
            species2 = seq2[j]
            
            match = " "
            
            if path == 0:
                i -= 1
                j -= 1
            elif path == 1:
                species1 = "-"
                j -= 1
            else:
                species2 = "-"
                i -= 1
            align1 = species1 + align1
            align2 = species2 + align2
            matches = match + matches
        
        return align1, align2


def merge_sequences(seq1, seq2):
    new_seq = ""
    for idx in range(len(seq2)):
        if seq1[idx] == '-':
            new_seq += seq2[idx]
        else:
            new_seq += seq1[idx]
    return new_seq


def multi_alignment(sequences):
    seq0, seq1 = sequences[:2]
    _, path_matrix = needleman_wunsch(seq0, seq1)
    merged_aligned, last_used_seq_aligend = get_alignment(path_matrix, seq0, seq1)
    print(f"{merged_aligned}\n{last_used_seq_aligend}")
    
    
    idx = 2
    for seq in sequences[1:]:
        merged_sequence = merge_sequences(merged_aligned, last_used_seq_aligend)
        _, path_matrix = needleman_wunsch(merged_sequence, seq)
        merged_aligned, last_used_seq_aligend = get_alignment(path_matrix, merged_sequence, seq)
        print(last_used_seq_aligend)
        idx += 1


def read_sequences(seq_filename):
    sequences = []
    with open(seq_filename, "r") as f:
        # Ignore first line
        next(f)
        seq = ""
        for ln in f:
            if ln.startswith(">"): 
                sequences.append(seq)
                seq = ""
                continue
            seq += ln.strip()
        sequences.append(seq)
    return  sequences


def main():
    seq_filename = sys.argv[1]
    second_argument = sys.argv[2]
    if second_argument == '--multi':
        sequences = read_sequences(seq_filename)
        multi_alignment(sequences=sequences)
    else:
        sequences = read_sequences(seq_filename)
        ind_seq1 = int(second_argument)
        ind_seq2 = int(sys.argv[3])
        seq1 = sequences[ind_seq1]
        seq2 = sequences[ind_seq2]

        _, path_matrix = needleman_wunsch(seq1, seq2)

        aligned_sequence1, aligned_sequence2 = get_alignment(path_matrix, seq1, seq2)
        print(f"{aligned_sequence1}\n{aligned_sequence2}")


if __name__ == '__main__':
    main()