from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
dataset = load_dataset("imdb")
dataset = dataset["train"].train_test_split(test_size=0.1)
train_data = dataset["train"]
val_data = dataset["test"]

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

# Metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Training config
training_args = TrainingArguments(
    output_dir="./scripts/models/final-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./scripts/models/final-model")
tokenizer.save_pretrained("./scripts/models/final-model")
