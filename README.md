# Transfer Equation
 
Запуск проекта:

python3 -m venv venv                                                # Создание виртуальной среды (ВС)
source venv/bin/activate                                            # Активация ВС
pip3 install requirements.txt                                       # Загрузка пакетов
mpiexec -n ProcNumber python3 -m mpi4py app.py                      # Запуск