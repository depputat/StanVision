import cv2
import pytesseract
import os
from tkinter import Tk, filedialog

def preprocess_image(image_path):
    """Подготовка изображения для лучшего OCR."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        gray = cv2.bilateralFilter(gray, 9, 75, 75)


        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return processed
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None


def extract_gcode_text(text):
    """Фильтрация текста, чтобы оставить только строки с G-кодом."""
    lines = text.splitlines()
    gcode_lines = []

    for line in lines:
        line = line.strip().upper()

        if any(token in line for token in ['G', 'M', 'X', 'Y', 'Z', 'F', 'S', 'E']):

            clean_line = ''.join(ch for ch in line if ch.isalnum() or ch in '.-+ ')
            if clean_line.strip():
                gcode_lines.append(clean_line)

    return '\n'.join(gcode_lines)


def recognize_gcode(image_path):
    """Основная функция — распознаёт G-код с картинки."""
    processed_image = preprocess_image(image_path)

    if processed_image is None:
        return ""

    try:
        # Распознаём текст с параметрами для кода
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=GXYZFMSE0123456789.-+'
        text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')

        # Фильтруем только G-код
        gcode = extract_gcode_text(text)
        return gcode
    except Exception as e:
        print(f"Ошибка при распознавании текста: {e}")
        return ""


def main():
    """GUI для выбора изображения и сохранения результата."""
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        image_path = filedialog.askopenfilename(
            title="Выбери изображение с G-кодом",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg")]
        )
        root.destroy()

        if not image_path:
            print("Файл не выбран.")
            return

        print(f"Обработка: {image_path}")
        gcode_text = recognize_gcode(image_path)

        if not gcode_text.strip():
            print("Не удалось распознать G-код. Попробуй другое изображение.")
            return

        output_path = os.path.splitext(image_path)[0] + "_recognized.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(gcode_text)

        print(f"Распознанный G-код сохранён в файл: {output_path}")
        #print(f"Распознано строк: {len(gcode_text.splitlines())}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()