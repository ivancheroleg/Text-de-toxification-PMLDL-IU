from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "t5-small-fine-tuned"
MODEL_PATH = "../../models"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "/" + MODEL_NAME)


def load_model():
    """
    Function for loading the model
    :return: tokenizer and model
    """

    # loading the model and run inference for it
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH + "/" + MODEL_NAME)
    model.eval()
    model.config.use_cache = False

    return tokenizer, model


def translate(model, inference_request, tokenizer=tokenizer):
    """
    Function for translation of the text
    :param model: given model
    :param inference_request: text to translate
    :param tokenizer: tokenizer for the model
    :return: translated text
    """

    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)


if __name__ == "__main__":
    tokenizer, model = load_model()
    print(translate(model, "This is a very disgusting sentence, trust me, most people would agree with me."))
