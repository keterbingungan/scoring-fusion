import regex
import string
import unicodedata
from collections import Counter
import logging

logger = logging.getLogger(__name__)




# Function to convert word numbers to integers and handle contractions
def word_transform(text):
    """Convert word numbers to integers and handle contractions."""
    
    WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
    contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
            "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", 
            "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", 
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", 
            "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", 
            "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", 
            "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", 
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", 
            "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", 
            "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", 
            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", 
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", 
            "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", 
            "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll", 
            "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", 
            "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", 
            "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 
            "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", 
            "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", 
            "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", 
            "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", 
            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", 
            "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", 
            "youll": "you'll", "youre": "you're", "youve": "you've"
        }
    
    # <answer> </answer> tags
    text = text.replace("<answer>", "").replace("</answer>", "")

    # Convert word numbers to integers
    for word, num in WORD_NUMBER_MAP.items():
        text = text.replace(word, str(num))
    
    # Handle contractions
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)
    
    return text




def _normalize(text):
    return unicodedata.normalize('NFD', text)


# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(word_transform(lower(s)))))



def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)



def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])



# Taken from HotPotQA Evaluation Script: https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall


