import easyocr
import ssl
import warnings
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning)


def text_regonition(file_path, text_file_name="result.txt"):
    reader = easyocr.Reader(["en"])
    result = reader.readtext(file_path, detail=0, paragraph=True)

    with open(text_file_name, "w") as file:
        for line in result:
            file.write(f"{line}\n\n")


    return f"Result wrote into {text_file_name}"

def main():
    file_path = input("Enter a file path: ")
    print(text_regonition(file_path=file_path, text_file_name="res.txt"))

if __name__ == "__main__":
    main()