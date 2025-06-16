from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

# Load model from local path
model_path = "./scripts/models/final-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load dataset
dataset = load_dataset("imdb")
val_data = dataset["train"].train_test_split(test_size=0.1)["test"]

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

val_data = val_data.map(tokenize_function, batched=True)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Dummy training args for eval-only use
args = TrainingArguments(output_dir="./eval_output")

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

results = trainer.evaluate()
print("📊 Evaluation Results:", results)
