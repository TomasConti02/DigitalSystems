#!/bin/bash

# Nome del file sorgente e del file di log
SOURCE_FILE="sumArr.cpp"
LOG_FILE="log.txt"

# Pulizia del file di log esistente
echo "Log dei test di esecuzione per $SOURCE_FILE" > $LOG_FILE
echo "======================" >> $LOG_FILE

# Opzioni di ottimizzazione da utilizzare
OPTIMIZATIONS=("O0" "O1" "O2" "O3")

# Ciclo sulle opzioni di ottimizzazione
for OPT in "${OPTIMIZATIONS[@]}"; do
    EXECUTABLE="sum_$OPT"  # Nome eseguibile per ogni livello di ottimizzazione

    # Compilazione con l'opzione corrente
    echo "Compilazione con -$OPT..."
    g++ -std=c++11 -$OPT $SOURCE_FILE -o $EXECUTABLE

    # Verifica se la compilazione è andata a buon fine
    if [ $? -ne 0 ]; then
        echo "Errore nella compilazione con -$OPT" >> $LOG_FILE
        continue
    fi

    # Scrivi l'intestazione per i risultati di questa ottimizzazione
    echo -e "\n\n--- Risultati per $EXECUTABLE con -$OPT ---" >> $LOG_FILE

    # Esegui il programma 20 volte e salva l'output nel file di log
    for i in {1..20}; do
        echo -e "\nEsecuzione #$i di $EXECUTABLE:" >> $LOG_FILE
        ./$EXECUTABLE >> $LOG_FILE
    done
done

echo "Tutti i test sono completi. I risultati sono salvati in $LOG_FILE"
