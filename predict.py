from train import Instructor
from config import DEFAULT_OPTION

if __name__ == '__main__':
    opt = DEFAULT_OPTION
    instrutor = Instructor(opt)
    # model_cpt = 'laptop_1638717418_cvt_epoch18_acc_65.26_f1_58.52.pkl'   # sup lap
    model_cpt = 'laptop_1638717634_cvt_epoch24_acc_69.33_f1_64.07.pkl'  # semi lap


    sentence = 'apple is aware of this issue and this is either old stock or a defective design involving the intel 4000 graphics chipset .'
    aspect = 'design'
    polarity = 1
    instrutor.predict(name=model_cpt,sample=(sentence,aspect,polarity))
