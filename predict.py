from train import Instructor
from config import DEFAULT_OPTION

if __name__ == '__main__':
    opt = DEFAULT_OPTION.set({'dataset': 'restaurant', 'clear_model': False})
    instrutor = Instructor(opt)

    # model_cpt = 'restaurant_semit.pkl'
    model_cpt = 'restaurant_semif.pkl'
    #
    sentence = 'the staff members are extremely friendly and even replaced my drink once when i dropped it outside . '
    aspect = 'drink'
    polarity = 0  # -1  neg:0 neu:1 pos:2 null:-1(unlabeled)
    instrutor.predict(name=model_cpt, sample=(sentence, aspect, polarity))
