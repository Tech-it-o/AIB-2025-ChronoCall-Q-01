from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_on_hub = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData-2"
try:
    model_loaded = AutoModelForCausalLM.from_pretrained(model_name_on_hub)
    tokenizer_loaded = AutoTokenizer.from_pretrained(model_name_on_hub)
    print("โมเดลถูกโหลดจาก Hugging Face Hub ได้สำเร็จแล้ว! ปัญหาได้รับการแก้ไขเรียบร้อย")
    # คุณสามารถลองทดสอบโมเดลของคุณได้ที่นี่
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")