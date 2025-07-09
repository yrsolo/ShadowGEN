from flask import Flask, request, jsonify, send_file, Response
import io
from PIL import Image
from ML_SERVER.sam import sam_process, pic2pil, pic2float
from ML_SERVER.processor import process_image

app = Flask(__name__)


def prepare_response(processed_images, processed_text, status=200):
    """
    Подготавливает ответ для возвращения из функции process.
    Возвращает изображение и текст в формате multipart/form-data.

    :param processed_image: объект PIL.Image, обработанное изображение
    :param processed_text: str, обработанный текст
    :return: Response объект Flask
    """
    # Сохраняем изображение в буфер
    # buffer = io.BytesIO()
    # processed_image.save(buffer, format="JPEG")
    # buffer.seek(0)

    # Формируем multipart-ответ
    boundary = "----CustomBoundaryString"
    response_body = []

    # Добавляем текст
    response_body.append(f"--{boundary}")
    response_body.append('Content-Disposition: form-data; name="message"')
    response_body.append("")
    response_body.append(processed_text)

    # Добавляем изображение
    for i, image in enumerate(processed_images):
        response_body.append(f"--{boundary}")
        response_body.append(f'Content-Disposition: form-data; name="image"; filename="processed_image_{i}.jpg"')
        response_body.append("Content-Type: image/jpeg")
        response_body.append("")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        response_body.append(buffer.getvalue())
    # response_body.append('TEST TEST TEST')

    # Закрываем boundary
    response_body.append(f"--{boundary}--")
    # response_body.append("")

    # Создаём HTTP-ответ
    response_body = b"\r\n".join(
        part if isinstance(part, bytes) else part.encode("utf-8") for part in response_body
    )
    # print(response_body)
    response = Response(
        response_body,
        content_type=f"multipart/form-data; boundary={boundary}",
        status=status,
    )

    return response

@app.route('/test', methods=['GET'])
def test():
    return 'OK'

@app.route('/process', methods=['POST'])
def process():
    # Получаем изображение из запроса
    if 'image' not in request.files:
        return prepare_response(None, 'Изображение не найдено', 400)

    print(request.form)

    image_file = request.files['image']
    image = Image.open(image_file)

    params = request.form

    # Здесь может быть ML-обработка
    # Например, обработка изображения (в данном случае просто возвращаем обратно)
    processed_images, text = process_image(image, params)
    # print(f'processed_image with shape {processed_image.size}')


    return prepare_response(processed_images, text, 200)

if __name__ == "__main__":
    pass
    app.run(debug=True, host="0.0.0.0", port=9001)
