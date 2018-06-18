GLOVE_DIR = "./glove.twitter.27B.100d.txt"
class_type = "AGR"
TRAIN_DATA_DIR ="./data_youtube/train_data/"+class_type+"/"
TEST_DATA_DIR ="./data_youtube/test_data/"+class_type+"/"
SAVE_PATH="./model_para/2CLSTM_c"+class_type+".h5"
RESULT_PATH="./model_para/2CLSTM_c" + class_type + "_train_test_details.txt"
AUGMENT=False
bs=32

