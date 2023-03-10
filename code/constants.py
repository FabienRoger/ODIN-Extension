NORM_BIAS = [125.3 / 255, 123.0 / 255, 113.9 / 255]
NORM_SCALE = [63.0 / 255, 62.1 / 255.0, 66.7 / 255.0]
EPS_FSGM = 1e-6


def file_name(nnName, dsName, algorithm, part):
    return f"./softmax_scores/confidence_{nnName}_{dsName}_{algorithm}_{part}.txt"
