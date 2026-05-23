# -*- coding: utf-8 -*-
"""
Script de Cómputo de Alto Desempeño (HPC) para Procesamiento de Imágenes.
Realiza la lectura, redimensionamiento, aplanamiento y consolidación de imágenes
en un dataset CSV utilizando paralelismo por procesos. Incluye módulo de Benchmarking.
"""

import os
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

# --- CONFIGURACIÓN GLOBAL ---
TARGET_SIZE = (16, 16)
CLASSES = ['A', 'B', 'C', 'D']
OUTPUT_FILE = "dataset_unificado.csv"

def procesar_clase(info_clase: Tuple[str, str]) -> str:
    """
    Procesa las imágenes de una clase específica de forma robusta.
    Ignora metadatos del OS y soporta rutas con caracteres especiales.
    
    Args:
        info_clase (Tuple[str, str]): Tupla que contiene la ruta base y la clase objetivo.
        
    Returns:
        str: Mensaje de registro (log) con los resultados y el tiempo de ejecución.
    """
    inicio_clase = time.perf_counter() 
    ruta_dataset, clase_objetivo = info_clase
    archivo_csv_clase = f"clase_{clase_objetivo}.csv"
    
    # Mapeo tolerante del nombre de la carpeta
    carpeta_real = None
    if os.path.exists(ruta_dataset):
        for nombre in os.listdir(ruta_dataset):
            if nombre.upper() == clase_objetivo.upper():
                carpeta_real = os.path.join(ruta_dataset, nombre)
                break
                
    if not carpeta_real or not os.path.isdir(carpeta_real):
        tiempo_clase = time.perf_counter() - inicio_clase
        return f"[WARN] Carpeta [{clase_objetivo}] no encontrada. (Tiempo: {tiempo_clase:.4f}s)"
        
    lineas = []
    
    for elemento in os.listdir(carpeta_real):
        # Filtro de archivos ocultos y metadatos del OS
        if elemento.startswith(('._', '.', 'Thumbs.db')):
            continue
            
        ruta_completa = os.path.join(carpeta_real, elemento)
        
        if os.path.isfile(ruta_completa):
            try:
                # Lectura binaria nativa para evadir problemas de decodificación con OpenCV (ej. caracteres especiales)
                with open(ruta_completa, "rb") as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            except Exception:
                continue
            
            if img is not None:
                re_size = cv2.resize(img, TARGET_SIZE)
                valores = re_size.ravel()
                cadena_pixeles = ",".join(valores.astype(str))
                lineas.append(f"{cadena_pixeles},{clase_objetivo}\n")
                
    tiempo_clase = time.perf_counter() - inicio_clase
            
    if lineas:
        with open(archivo_csv_clase, "w", encoding="utf-8") as f:
            f.writelines(lineas)
        return f"[SUCCESS] Clase {clase_objetivo}: {len(lineas):4} imágenes procesadas en {tiempo_clase:.4f}s."
        
    return f"[INFO] Clase {clase_objetivo}: Sin datos válidos. (Tiempo: {tiempo_clase:.4f}s)"

def unificar_csvs(clases: List[str], archivo_final: str) -> None:
    """
    Une los archivos CSV individuales en un único dataset consolidado.
    
    Args:
        clases (List[str]): Lista de las clases procesadas.
        archivo_final (str): Nombre del archivo de salida consolidado.
    """
    print("\n[INFO] Unificando archivos CSV individuales...")
    con_datos = False
    
    with open(archivo_final, "w", encoding="utf-8") as destino:
        for c in clases:
            archivo_origen = f"clase_{c}.csv"
            if os.path.exists(archivo_origen):
                with open(archivo_origen, "r", encoding="utf-8") as origen:
                    destino.write(origen.read())
                os.remove(archivo_origen)  # Limpieza de archivos temporales
                con_datos = True
    
    if con_datos:
        print(f"[SUCCESS] Dataset unificado guardado exitosamente como: '{archivo_final}'")
    else:
        print("[WARN] No se generó el dataset final debido a la falta de datos procesados.")

def main() -> None:
    """Función principal que orquesta el Benchmarking y el procesamiento de datos."""
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_dataset = os.path.join(directorio_script, 'Dataset')
    
    tareas = [(ruta_dataset, c) for c in CLASSES]
    num_nucleos = min(os.cpu_count(), len(CLASSES))
    
    print("=" * 70)
    print(" INICIANDO BENCHMARK: SECUENCIAL VS PARALELO ".center(70, "="))
    print("=" * 70)
    
    # --- FASE 1: PROCESAMIENTO SECUENCIAL ---
    print("\n[FASE 1] Procesamiento Secuencial (1 Hilo)")
    inicio_sec = time.perf_counter()
    
    for tarea in tareas:
        res = procesar_clase(tarea)
        print("  " + res)
        
    fin_sec = time.perf_counter()
    tiempo_secuencial = fin_sec - inicio_sec
    print("-" * 70)
    print(f"  Tiempo Total Secuencial: {tiempo_secuencial:.4f} segundos")
    
    # --- FASE 2: PROCESAMIENTO PARALELO ---
    print(f"\n[FASE 2] Procesamiento Paralelo ({num_nucleos} Hilos)")
    inicio_par = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=num_nucleos) as executor:
        resultados = executor.map(procesar_clase, tareas)
        for res in resultados:
            print("  " + res)
            
    fin_par = time.perf_counter()
    tiempo_paralelo = fin_par - inicio_par
    print("-" * 70)
    print(f"  Tiempo Total Paralelo: {tiempo_paralelo:.4f} segundos\n")
    
    # --- FASE 3: CONSOLIDACIÓN ---
    unificar_csvs(CLASSES, OUTPUT_FILE)
    
    # --- RESULTADOS TÉCNICOS ---
    print("\n" + "=" * 70)
    print(" RESULTADOS DEL RENDIMIENTO (SPEEDUP) ".center(70, "="))
    print("=" * 70)
    print(f"  Carga Secuencial: {tiempo_secuencial:.4f} s")
    print(f"  Carga Paralela:   {tiempo_paralelo:.4f} s")
    
    if tiempo_paralelo < tiempo_secuencial:
        aceleracion = tiempo_secuencial / tiempo_paralelo
        mejora_porcentual = ((tiempo_secuencial - tiempo_paralelo) / tiempo_secuencial) * 100
        print(f"  Aceleración (Speedup): {aceleracion:.2f}x")
        print(f"  Reducción de tiempo:   {mejora_porcentual:.1f}%")
    else:
        print("  [WARN] Anomalía detectada: El procesamiento secuencial fue más rápido.")
    print("=" * 70)

if __name__ == '__main__':
    main()