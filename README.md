# Entregable 2: análisis modal con Streamlit

Aplicación educativa para analizar un edificio de corte con varios niveles.
Permite editar masas, alturas y rigideces, cargar esos datos desde un TXT,
comparar los modos, visualizar el esquema referencial por nivel y exportar un
informe PDF. El esquema usa las alturas, rigideces y pesos derivados de las
masas ingresadas manualmente o importadas del TXT.

## Estructura del proyecto

Todos los archivos del entregable están dentro de la carpeta `ENTREGABLE 2`.
La interfaz, los cálculos y los gráficos están separados para que cada parte
del programa sea fácil de identificar:

```text
ENTREGABLE 2/
├── app.py                       # Página web e interfaz Streamlit
├── src/
│   └── analisis_modal/
│       ├── __init__.py           # Funciones públicas que importa la página
│       ├── analysis.py           # Cálculos de matrices, modos y respuesta
│       ├── data_io.py            # Lectura TXT, validación y unidades
│       ├── plots.py              # Creación de los gráficos
│       └── report.py             # Generación del informe PDF
├── data/
│   └── caso_referencia.txt       # Datos de ejemplo del ejercicio
├── tests/                        # Pruebas del análisis y la entrada TXT
├── infografia_entregable2.png    # Infografía lista para compartir o imprimir
├── infografia_entregable2.svg    # Infografía vectorial editable
├── requirements.txt              # Dependencias de Python
├── .gitignore                    # Excluye entornos y archivos locales
└── README.md                     # Instrucciones y documentación
```

`.venv/` es un entorno virtual local para ejecutar la aplicación. No contiene
código fuente y está excluido del repositorio mediante `.gitignore`.

La infografía [PNG](./infografia_entregable2.png) resume la operación del
programa y muestra resultados del caso de referencia con los parámetros
sísmicos iniciales indicados en la página. También está disponible en
[formato vectorial SVG](./infografia_entregable2.svg) para ampliar o editar
sin pérdida de calidad.

## Inicio

Desde esta carpeta, instala las dependencias y ejecuta:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si el comando `py` no está disponible, reemplázalo por `python`.

La carpeta `src/analisis_modal` contiene las funciones reutilizables; `app.py`
es únicamente la interfaz web y `tests/` contiene pruebas del cálculo y del
formato de entrada.

## Datos y unidades

Los pisos se enumeran de abajo hacia arriba. En la tabla manual, cada fila
representa un nivel e incluye altura de entrepiso, masa y rigidez de ese
entrepiso. Se puede cargar el ejemplo `data/caso_referencia.txt` o usar el
botón de descarga de plantilla de la página.

El TXT debe llevar encabezados y usar punto decimal. Se aceptan separadores
coma, punto y coma o tabulador. Ejemplo:

```text
Piso;Altura;Masa;Rigidez
1;11;2.16;108
2;11;1.92;90
3;11;1.68;60
```

Se reconocen encabezados en español o inglés (`Piso/Level`, `Altura/Height`,
`Masa/Mass`, `Rigidez/Stiffness`); la columna de piso es opcional. Los números
se interpretan en las unidades elegidas en la aplicación:

| Sistema | Altura | Masa | Rigidez | Fuerza |
| --- | --- | --- | --- | --- |
| Métrico | m | kg | kN/m | kN |
| Imperial | in | kip·s²/in | kip/in | kip |

Al cambiar de sistema, los datos de la tabla se convierten para conservar el
mismo modelo físico.

## Método y alcance

Se construye la matriz de rigidez de un edificio de corte a partir de las
rigideces de entrepiso y una matriz de masa diagonal. El problema generalizado
de autovalores entrega frecuencias, periodos y formas modales. Se calculan los
factores y masas modales efectivos, desplazamientos espectrales, derivas,
fuerzas y cortantes por nivel, aceleraciones y su combinación SRSS.

El espectro de diseño se calcula con la forma de la Norma E.030 peruana. El
usuario ingresa los factores `Z`, `U`, `S`, `R` y los periodos de esquina `Tp`
y `TL`. Se emplea `Sa = Z·U·S·C(T)·g/R`, con `C(T) = 2.5` para `T ≤ Tp`,
`C(T) = 2.5·Tp/T` para `Tp < T ≤ TL` y
`C(T) = 2.5·Tp·TL/T²` para `T > TL`. El gráfico presenta `Sa` en unidades de
gravedad (`g`); los cálculos modales usan la aceleración en las unidades del
sistema elegido. Comprueba y proporciona los parámetros aplicables al sitio,
uso y perfil de suelo de cada proyecto.

El ejemplo imperial inicial usa tres niveles de 11 in, masas de
2.16/1.92/1.68 kip·s²/in y rigideces de 108/90/60 kip/in, de abajo hacia
arriba. Los parámetros sísmicos iniciales de la interfaz son los ilustrados
en la imagen de referencia (`Z=0.35`, `U=1`, `S=1.15`, `R=8`, `Tp=0.6 s`,
`TL=2 s`) y se deben reemplazar según el proyecto.

Este programa es una herramienta de aprendizaje y no sustituye una revisión
profesional ni la aplicación de los requisitos de diseño sísmico vigentes.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
