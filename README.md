# Language and Mental Health

## Abstract
Questo progetto esplora la relazione tra salute mentale e linguaggio, concentrandosi su sei disturbi: depressione, comportamento suicidario, ansia, disturbo bipolare, stress e disturbo della personalità. L'obiettivo principale è identificare eventuali pattern linguistici specifici di questi disturbi, e confrontarli con gli enunciati classificati nel dataset come prodotti da individui neurotipici. I risultati verranno analizzati alla luce delle principali pubblicazioni nei campi medico e psicolinguistico, come indicato nelle References del report.

## Dataset
Abbiamo selezionato un dataset da Kaggle, che unisce nove dataset individuali contenenti oltre 53.000 enunciati di utenti di social media (inclusi Reddit e Twitter). Questi enunciati sono stati etichettati in una delle sette categorie, in base allo stato mentale riflesso nelle parole utilizzate: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress e Personality disorder.

## Motivazioni del nostro progetto
L'idea alla base del progetto nasce dal nostro comune interesse per la salute mentale, un tema che sta guadagnando sempre maggiore attenzione. Il linguaggio è un importante indicatore dello stato mentale degli individui e, osservando le parole da loro frequentemente utilizzate, possiamo cogliere segnali del loro stato d'animo. Inoltre, avendo recentemente acquisito conoscenze sul trattamento automatico delle lingue (NLP) durante il primo anno di laurea magistrale in Linguistica, siamo rimaste particolarmente colpite dai task di sentiment analysis ed emotion detection, strumenti ormai utilizzati da linguisti e studiosi del linguaggio in vari ambiti. Crediamo che i risultati ottenuti possano costituire una base significativa per studi futuri, in particolare nella valutazione della salute mentale attraverso l'analisi linguistica.

## Struttura del codice
Per una lettura più fluida, abbiamo così impostato il codice: all'inizio sono presenti le analisi svolte sul dataset, successivamente si trovano i codici per la creazione e la visualizzazione dei grafici e, infine, vengono eseguiti i test delle funzioni precedentemente definite.

## Installazione e utilizzo
Per riprodurre il progetto, seguire questi passaggi:
Assicurarsi di avere installato Python 3.11.5.

Installare le dipendenze richieste eseguendo:  
	`pip install -r requirements.txt`

Una volta installate le dipendenze, eseguire il codice con il seguente comando:  
	`python code.py`

Seguendo i sopracitati passaggi, verranno generati localmente i grafici in PNG, i file CSV e TXT. 

## File e cartelle del progetto
- code.py: Contiene il codice per l'analisi del testo e le visualizzazioni.  
- requirements.txt: Elenca le librerie necessarie per il progetto.  
- combined_data: Contiene il dataset utilizzato per l'analisi.  
- plots\: Contiene i grafici generati durante il progetto.
- outputs\: Contiene i risultati numerici delle analisi. 
