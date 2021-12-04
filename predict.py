from train import Instructor
from config import DEFAULT_OPTION

if __name__ == '__main__':
    opt = DEFAULT_OPTION
    instrutor = Instructor(opt)
    model_cpt = 'laptop_1638529622_PositionEncoder_epoch19_acc_75.13_f1_71.37.pkl'
    sentence = 'apple is aware of this issue and this is either old stock or a defective design involving the intel 4000 graphics chipset .'
    aspect = 'design'
    polarity = 1
    instrutor.predict(name=model_cpt,sample=(sentence,aspect,polarity))
