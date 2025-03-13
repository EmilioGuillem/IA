# from numba import jit, cuda
import numpy as np
from datasets import load_dataset
import os
import torch
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from trl import SFTTrainer

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

def main():
    path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\audio_cassifier'
    path_to_hggf_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\audio_classifier'
    # Check if CUDA is available
    print(torch.cuda.is_available()) # True if CUDA is available

    # Get the number of GPUs available
    print(torch.cuda.device_count()) # Number of GPUs

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(0)) #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    #Load the processor and model.
    MODEL_NAME="carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    #Load the dataset
    from datasets import load_dataset, Audio
    import evaluate
    ds=load_dataset("ciempiess/ciempiess_test", split="test")
    # dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context.json",encoding='latin1',  split=['train[:80%]', 'train[80%:]'])
    #Downsample to 16kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Process the dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        #Batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["normalized_text"]).input_ids
        return batch
    ds = ds.map(prepare_dataset, remove_columns=ds.column_names,num_proc=1)

    #Define the evaluation metric
    import numpy as np
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        #We do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    #Do the evaluation (with batch_size=1)
    model = model.to(torch.device("cuda"))
    def map_to_result(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
        return batch
    results = ds.map(map_to_result,remove_columns=ds.column_names)

    #Compute the overall WER now.
    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["sentence"])))


    model.save_pretrained(path_to_save_model)
    processor.save_pretrained(path_to_save_model)

    #save gguf for ollama serve
    # Step 3: Save the Fine-Tuned Model

    # After training, you can save the fine-tuned model:

    # model.save_pretrained("C:/Users/Emilio Guillem/Documents/GIT/IA/src/llm/fine_tuned_llama")
    # tokenizer.save_pretrained("C:/Users/Emilio Guillem/Documents/GIT/IA/src/llm/fine_tuned_llama")

    # Notes:
    # Dataset: Replace "path_to_your_dataset" with the actual path or name of your dataset.
    # Hyperparameters: Adjust the hyperparameters (e.g., learning rate, batch size, number of epochs) as needed for your specific use case.
    # Hardware: Ensure you have the necessary hardware (e.g., GPU) to handle the fine-tuning process, especially for large models like LLaMA 3.2B.

    # Feel free to adapt this script to better fit your needs. Happy fine-tuning!
    # import os
    # os.environ['HF_TOKENIZER'] = "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"
    # model.push_to_hub("Emiliogs/orbital", use_auth_token=os.getenv("HF_TOKEN"))
    # tokenizer.push_to_hub("Emiliogs/orbital", use_auth_token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()