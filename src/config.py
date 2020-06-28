# configs

# train files
TRAIN_AUTHOR_PATH = '../dataset/train_author.json'
TRAIN_PUB_PATH = '../dataset/train_pub.json'
TRAIN_RESULT_PATH = '../dataset/train_res.json'

# valid files
VAL_AUTHOR_PATH = '../dataset/valid/sna_valid_author_raw.json'
VAL_PUB_PATH = '../dataset/valid/sna_valid_pub.json'
VAL_RESULT_PATH = '../dataset/valid/valid_res.json'

# test_files
TEST_AUTHOR_PATH = '../dataset/test/sna_test_author_raw.json'
TEST_PUB_PATH = '../dataset/test/test_pub_sna.json'
TEST_RESULT_PATH = '../dataset/test/example_evaluation_scratch.json'

ALL_TEXT_PATH = '../dataset/all_text.txt'
WORD_EMBEDDING_MODEL_PATH = '../model/word_embedding.model'
STOP_WORDS_PATH = '../dataset/stop_words.txt'

# feature files
TRAIN_PUB_FEATURES_PATH = '../dataset/features/train_pub_features.txt'
TRAIN_SEMANTIC_FEATURES_PATH = '../dataset/features/train/semantic/'
TRAIN_SEMANTIC_FEATURES_PATH_64 = '../dataset/features/train/semantic64/'
TRAIN_RELATION_FEATURES_PATH = '../dataset/features/train/relation/'

VAL_PUB_FEATURES_PATH = '../dataset/features/val_pub_features.txt'
VAL_SEMANTIC_FEATURES_PATH = '../dataset/features/val/semantic/'
VAL_SEMANTIC_FEATURES_PATH_64 = '../dataset/features/val/semantic64/'
VAL_RELATION_FEATURES_PATH_DIS = '../dataset/features/val/val_dis/'
VAL_RELATION_FEATURES_PATH_GEN = '../dataset/features/val/val_gen/'

# similary of semantic features by Author
TRAIN_SIMI_SENMATIC_FOLDER = '../dataset/similarity/semantic/train/'
VAL_SIMI_SENMATIC_FOLDER = '../dataset/similarity/semantic/val/'

# similary of relation features by Author
TRAIN_SIMI_RELATION_FOLDER = '../dataset/similarity/relation/train/'
VAL_SIMI_RELATION_FOLDER = '../dataset/similarity/relation/val/'

TRIPLETS_PATH = '../dataset/triplets/'

TRAIN_GRAPH_PATH = '../dataset/graph/train/'
VAL_GRAPH_PATH = '../dataset/graph/val/'

# HeGAN training hyperparameters

batch_size = 32
lambda_gen = 1e-5
lambda_dis = 1e-5
n_sample = 16
lr_gen = 0.0001  # 1e-3
lr_dis = 0.0001  # 1e-4
n_epoch = 20
saves_step = 10
sig = 1.0
label_smooth = 0.0
d_epoch = 15
g_epoch = 5
n_emb = 100
emb_filenames_gen = '../results/gen/'
emb_filenames_dis = '../results/dis/'

model_log = '../model_log/'

# model path
SEMANTIC_MODEL_PATH = '../model/semantic_model.pkl'