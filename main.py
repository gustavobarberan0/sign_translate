#!/usr/bin/env python3
"""
SignTranslate - Traductor de Lenguaje de Señas Español
Punto de entrada principal de la aplicación
"""

import tkinter as tk
from src.app import SignLanguageApp

def main():
    """Función principal que inicia la aplicación"""
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()