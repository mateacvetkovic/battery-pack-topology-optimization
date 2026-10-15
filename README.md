# Program za određivanje optimalnog spoja baterijskih članaka

Programsko rješenje razvijeno u sklopu diplomskog rada za automatsko generiranje, konfiguraciju i optimizaciju rasporeda ćelija u baterijskim paketima nepravilnih geometrijskih oblika (pravokutnici i trokuti) s primjenom heurističkih metoda i umjetnog imunološkog algoritma (CLONALG).


## Preduvjeti

Prije pokretanja programa potrebno je imati instaliran **Python** (preporučena vrezija 3.12).
Kako bi se omogućio prikaz interaktivnih prozora na Linuxu, potrebno je prije kreiranja virtualnog okruženja instalirati sistemski paket `python3-tk`:
    sudo apt update
    sudo apt install python3-tk

## Upute za instalaciju i pokretanje

### 1. Preuzimanje i pozicioniranje u mapu projekta
Otvorite terminal i pređite u mapu u kojoj se nalazi izvorni kod programa i datoteka `requirements.txt`.

### 2. Kreiranje i aktivacija virtualnog okruženja (venv)
Preporučuje se izrada izoliranog virtualnog okruženja:
Linux:
    python3 -m venv venv
    source venv/bin/activate
Windows:
bash
    python -m venv venv
    venv\Scripts\activate
### 3. Instalacija potrebnih biblioteka
Nakon aktivacije okruženja, instalirajte sve potrebne zavisnosti:
    pip install -r requirements.txt 
### 4. Pokretanje programa
Glavna ulazna točka programa pokreće se naredbom:
    python3 main.py


## Rad s programom
Nakon pokretanja skripte, u terminalu će se pojaviti izbornik kroz koji se unose tip ćelije, format ćelije (18650 ili 21700), udaljenost među ćelijama (mm), tip mreže (saćasta/pravokutna), oblik paketa (pravokutnik/trokut), duljine stranica paketa (cm).
Program zatim iscrtava paket i omogućuje ručno uklanjanje dijela ćelija za BMS unosom njihovih ID-jeva sa slike u terminal. Za zadani paket program ispisuje moguće konfiguracije, nakon čega korisnik unosi željeni napon paketa. 
Po završetku unosa, program pokreće optimizacijski algoritam, ispisuje ocjenu kvalitete (fitness) i parametre spojeva u terminal te automatski sprema grafički prikaz konačnog dizajna u obliku .png slika u radnu mapu projekta. S obzirom da je moguće da postoji više najboljih rješenja (različita konfiguracija, ista dobrota), sve se slike konfiguracija pohranjuju u mapu, a samo jedna od njih otvara u prozoru za prikaz. Program završava s radom i zatvara sve prozore pritiskom tipke ENTER.
