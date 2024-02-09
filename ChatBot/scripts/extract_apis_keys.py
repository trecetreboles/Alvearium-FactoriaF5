import os
from dotenv import load_dotenv, find_dotenv

result = load_dotenv(find_dotenv(), override=True)

def load(result):
    if result:
        print("Variables de entorno cargadas exitosamente.")
    else:
        print("No se pudo cargar el archivo .env.")

        # Crear el archivo .env con las variables deseadas si no existe
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY=MY_OPENAI_API_KEY\n")

        print("Se creó el archivo .env con las variables iniciales.")
        # Vuelve a cargar las variables de entorno después de crear el archivo
        load_dotenv()

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY") or "MY_OPENAI_API_KEY"

    return os.environ['OPENAI_API_KEY']