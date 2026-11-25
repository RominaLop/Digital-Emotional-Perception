import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Configuración ----------------
MODELO = "face_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
CARPETA_IMAGENES = "imagenes"

FREEZE_SEGUNDOS = 3.0      # cuánto tiempo se queda congelado
FRAMES_CONFIRMAR = 6       # frames seguidos con el mismo gesto para aceptarlo
ANCHO_PANEL, ALTO_PANEL = 640, 480

# ---------------- Gestos ----------------
# Cada métrica sale de uno o varios blendshapes de MediaPipe (promedio izq/der)
METRICAS = {
    "cejas_internas": ["browInnerUp"],
    "cejas_externas": ["browOuterUpLeft", "browOuterUpRight"],
    "ceno":           ["browDownLeft", "browDownRight"],
    "boca_abierta":   ["jawOpen"],
    "sonrisa":        ["mouthSmileLeft", "mouthSmileRight"],
    "comisuras_abajo": ["mouthFrownLeft", "mouthFrownRight"],
    "labio_inf_arriba": ["mouthShrugLower"],
    "beso":           ["mouthPucker"],
    "ojos_cerrados":  ["eyeBlinkLeft", "eyeBlinkRight"],
}


def clasificar(m):
    """Devuelve el nombre del gesto según las métricas (el orden importa)."""
    if m["ceno"] > 0.10:
            return "ENOJO"
    if m["beso"] > 0.97:
        return "BESO"
    if m["labio_inf_arriba"] > 0.60 and m["sonrisa"] < 0.30:
                return "TRISTEZA"
    if m["cejas_internas"] > 0.60 and m["cejas_externas"] > 0.60:
        return "SORPRESA"
    if m["sonrisa"] > 0.75 and m["boca_abierta"] < 0.30:
            return "APROBACION"
    if m["boca_abierta"] > 0.40 and m["sonrisa"] > 0.60:
        return "EMOCION"
    if m["ojos_cerrados"] > 0.60:
        return "DORMIDO"
    return "SIN GESTO"


# ---------------- Utilidades ----------------
def descargar_modelo():
    if not os.path.exists(MODELO):
        print("Descargando modelo de MediaPipe...")
        urllib.request.urlretrieve(URL_MODELO, MODELO)


def ajustar_cover(img, w, h):
    """Redimensiona y recorta la imagen para llenar w x h sin deformarla."""
    ih, iw = img.shape[:2]
    escala = max(w / iw, h / ih)
    nueva = cv2.resize(img, (int(iw * escala), int(ih * escala)))
    nh, nw = nueva.shape[:2]
    x, y = (nw - w) // 2, (nh - h) // 2
    return nueva[y:y + h, x:x + w]


def panel_texto(texto, w=ANCHO_PANEL, h=ALTO_PANEL):
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(panel, texto, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def cargar_imagenes():
    imagenes = {}
    for gesto in ["APROBACION", "TRISTEZA", "DORMIDO", "BESO",
                  "EMOCION", "SORPRESA", "ENOJO"]:
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            ruta = os.path.join(CARPETA_IMAGENES, gesto.lower() + ext)
            if os.path.exists(ruta):
                img = cv2.imread(ruta)
                break
        if img is None:
            print(f"Aviso: no encontré imagen para {gesto}, uso un texto.")
            imagenes[gesto] = panel_texto(gesto)
        else:
            imagenes[gesto] = ajustar_cover(img, ANCHO_PANEL, ALTO_PANEL)
    return imagenes


def dibujar_contornos(frame, landmarks):
    h, w = frame.shape[:2]
    for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        a, b = landmarks[c.start], landmarks[c.end]
        cv2.line(frame, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)),
                 (255, 255, 255), 1, cv2.LINE_AA)


def dibujar_info(frame, gesto, metricas):
    cv2.putText(frame, gesto, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)
    for i, (nombre, valor) in enumerate(metricas.items()):
        cv2.putText(frame, f"{nombre}: {valor:.2f}", (12, 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def dibujar_freeze(pantalla, restante):
    """Barra de tiempo restante del congelado."""
    h, w = pantalla.shape[:2]
    progreso = max(0.0, restante / FREEZE_SEGUNDOS)
    cv2.rectangle(pantalla, (0, h - 10), (int(w * progreso), h), (0, 200, 255), -1)
    cv2.putText(pantalla, f"FREEZE {restante:.1f}s  (ESPACIO para seguir)",
                (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


# ---------------- Programa principal ----------------
def main():
    descargar_modelo()
    imagenes = cargar_imagenes()
    panel_vacio = panel_texto("HAZ UNA EXPRESION")

    opciones = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELO),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    candidato, contador = "SIN GESTO", 0
    ultimo_disparado = None          # evita repetir el mismo gesto sin volver a neutral
    congelado_hasta = 0.0
    pantalla_congelada = None
    inicio = time.time()

    with vision.FaceLandmarker.create_from_options(opciones) as detector:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            ahora = time.time()

            # ----- Modo FREEZE: mostrar la imagen congelada -----
            if ahora < congelado_hasta:
                pantalla = pantalla_congelada.copy()
                dibujar_freeze(pantalla, congelado_hasta - ahora)
                cv2.imshow("Expresiones", pantalla)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord(" "):
                    congelado_hasta = 0.0
                continue

            # ----- Detección normal -----
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (ANCHO_PANEL, ALTO_PANEL))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect_for_video(imagen_mp, int((ahora - inicio) * 1000))

            gesto = "SIN GESTO"
            metricas = {k: 0.0 for k in METRICAS}
            if res.face_landmarks:
                dibujar_contornos(frame, res.face_landmarks[0])
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                metricas = {k: sum(bs.get(n, 0) for n in nombres) / len(nombres)
                            for k, nombres in METRICAS.items()}
                gesto = clasificar(metricas)

            # Confirmar que el gesto se mantiene unos frames (evita parpadeos)
            if gesto == candidato:
                contador += 1
            else:
                candidato, contador = gesto, 1
            confirmado = contador >= FRAMES_CONFIRMAR

            if confirmado and candidato == "SIN GESTO":
                ultimo_disparado = None

            dibujar_info(frame, gesto, metricas)
            derecha = panel_vacio

            if confirmado and candidato != "SIN GESTO" and candidato != ultimo_disparado:
                derecha = imagenes[candidato]
                pantalla_congelada = np.hstack([frame, derecha])
                congelado_hasta = ahora + FREEZE_SEGUNDOS
                ultimo_disparado = candidato
                contador = 0

            cv2.imshow("Expresiones", np.hstack([frame, derecha]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()