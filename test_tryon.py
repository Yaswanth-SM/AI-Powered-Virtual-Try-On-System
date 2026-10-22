
from gradio_client import Client, handle_file

client = Client("yisol/IDM-VTON")

result = client.predict(
    dict={
        "background": handle_file("test_images/person.jpg"),
        "layers": [],
        "composite": None,
    },
    garm_img=handle_file("test_images/shirt.jpg"),
    garment_des="T-shirt",
    is_checked=True,
    is_checked_crop=False,
    denoise_steps=30,
    seed=42,
    api_name="/tryon",
)

print("Result received:")
print(result)
