from stardist.models import StarDist3D

model = StarDist3D.from_pretrained('3D_demo')

#save model to folder
print(model.base_dir)