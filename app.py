import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import warnings
warnings.filterwarnings("ignore")

# Load the fine-tuned model and tokenizer
model_path = "t5_finetuned"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_summary(text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = input_ids.to(device)

    summary_ids = model.generate(input_ids, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

iface = gr.Interface(
    fn=generate_summary,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to summarize..."),
    outputs=gr.Textbox(label="Generated Summary"),
    title="T5 Text Summarization",
    description="Enter a long text, and the T5 model will summarize it for you.",
    theme="default",
)
iface.launch()