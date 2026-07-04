from transformers import VideoMAEImageProcessor, VideoMAEModel

model_name = "MCG-NJU/videomae-base-finetuned-kinetics"

processor = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEModel.from_pretrained(model_name)