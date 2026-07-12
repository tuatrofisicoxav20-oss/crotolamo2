"""Panel de control de escritorio de Crotolamo, partido en módulos.

- systemd:  interacción con systemctl/journalctl (sin gi, testeable headless)
- config:   env file de modo + canal de pausa de escucha (sin gi)
- terminal: detección de emulador de terminal (sin gi)
- window:   Panel (GTK)
- app:      App y main()
"""
