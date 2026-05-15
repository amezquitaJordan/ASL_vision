"""
ARCHIVO: motion_letters.py
MÓDULO: Detección de Movimiento
DESCRIPCIÓN: Detector temporal simple para los movimientos de las letras J y Z del ASL.
PARTE DE LA APP QUE CONTROLA: Análisis de la trayectoria de la mano para reconocer letras en movimiento.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Deque

# Define un punto como una tupla de coordenadas X e Y
Point = tuple[float, float]


@dataclass
class MotionLetterDetector:
    """
    Clase: Detecta las letras J y Z a partir de las trayectorias normalizadas de la punta del dedo índice.
    """

    max_points: int = 18          # Cantidad máxima de puntos en la memoria
    min_points: int = 6           # Cantidad mínima de puntos requeridos para analizar
    min_distance: float = 0.18    # Distancia mínima que debe recorrerse
    cooldown_frames: int = 12     # Fotogramas de espera luego de una detección exitosa
    points: Deque[Point] = field(default_factory=deque)  # Historial de puntos
    _cooldown: int = 0            # Contador de enfriamiento (cooldown)

    def update(self, point: Point | None) -> str | None:
        """
        Función: Añade un punto al historial y devuelve "J" o "Z" cuando se reconoce el movimiento.
        Parámetros: point (Point | None) - Las coordenadas de la punta del dedo.
        """
        # Si no hay punto, reinicia el estado
        if point is None:
            self.reset()
            return None

        # Si el contador de espera está activo, lo disminuye
        if self._cooldown > 0:
            self._cooldown -= 1

        # Agrega el punto actual a la cola de puntos
        self.points.append(point)
        
        # Limita el tamaño de la cola eliminando los puntos más antiguos
        while len(self.points) > self.max_points:
            self.points.popleft()

        # No analiza si está en espera o si no tiene suficientes puntos
        if self._cooldown > 0 or len(self.points) < self.min_points:
            return None

        # Convierte la cola a lista para evaluar el trazado
        path = list(self.points)
        
        # Verifica si el patrón corresponde a una "Z"
        if self._detect_z(path):
            self._trigger()
            return "Z"
            
        # Verifica si el patrón corresponde a una "J"
        if self._detect_j(path):
            self._trigger()
            return "J"
            
        return None

    def reset(self) -> None:
        """
        Función: Limpia el historial de puntos y el tiempo de espera.
        """
        self.points.clear()
        self._cooldown = 0

    def _trigger(self) -> None:
        """
        Función: Limpia los puntos y activa el tiempo de espera (cooldown) tras una detección.
        """
        self.points.clear()
        self._cooldown = self.cooldown_frames

    def _path_distance(self, path: list[Point]) -> float:
        """
        Función: Calcula la distancia geométrica total recorrida por todos los puntos de la trayectoria.
        """
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))

    def _detect_j(self, path: list[Point]) -> bool:
        """
        Función: Analiza la trayectoria para ver si cumple con los criterios de la letra "J".
        Un trazo hacia abajo seguido de una curva hacia la izquierda.
        """
        start = path[0]
        end = path[-1]
        total_distance = self._path_distance(path)
        
        # Condición: El movimiento general debe ser hacia abajo
        downward = end[1] - start[1] > self.min_distance
        # Condición: El movimiento final debe tener un gancho hacia la izquierda
        hook_left = start[0] - end[0] > 0.12
        # Condición: La trayectoria debe presentar una curvatura izquierda a derecha
        has_curve = min(point[0] for point in path[-3:]) < min(point[0] for point in path[:3])
        
        return total_distance > self.min_distance and downward and hook_left and has_curve

    def _detect_z(self, path: list[Point]) -> bool:
        """
        Función: Analiza la trayectoria para ver si cumple con los criterios de la letra "Z".
        Tres trazos: horizontal, diagonal izquierda inferior, horizontal.
        """
        if len(path) < 6 or self._path_distance(path) < self.min_distance * 1.8:
            return False

        # Divide la trayectoria en 3 segmentos: inicio, medio y fin
        first = path[: len(path) // 3]
        middle = path[len(path) // 3 : 2 * len(path) // 3]
        last = path[2 * len(path) // 3 :]
        
        if not first or not middle or not last:
            return False

        # Diferencias (deltas) en los 3 segmentos
        first_dx = first[-1][0] - first[0][0]
        first_dy = abs(first[-1][1] - first[0][1])
        mid_dx = middle[-1][0] - middle[0][0]
        mid_dy = middle[-1][1] - middle[0][1]
        last_dx = last[-1][0] - last[0][0]
        last_dy = abs(last[-1][1] - last[0][1])

        # Condiciones características del movimiento 'Z'
        top_horizontal = first_dx > 0.15 and first_dy < 0.10
        diagonal_down_left = mid_dx < -0.08 and mid_dy > 0.12
        bottom_horizontal = last_dx > 0.15 and last_dy < 0.12
        
        return top_horizontal and diagonal_down_left and bottom_horizontal
