import json
import os
import math
import random
import numpy as np
import openrouteservice
import matplotlib.pyplot as plt
import contextily as ctx
from geopy.distance import geodesic
import time


api_key = os.getenv("OPENROUTE_KEY")
openrouteservice_client = openrouteservice.Client(key=api_key)

# Pozadinska mapa se preuzima samo jednom i čuva kao GeoTIFF (EPSG:3857).
# Svako sledeće pokretanje čita mapu iz fajla, bez pristupa internetu.

# Granice mape u stepenima: centralni Beograd, od Novog Beograda (Delta City)
# do Karaburme i od Dorćola do Banjice. Sve lokacije iz programa su unutra,
# sa oko 2 km rezerve do ivice mape.
# Za Zemun stavite "zapad": 20.37, "istok": 20.54, "sever": 44.86 (~290 pločica).
GRANICE_BEOGRADA = {"zapad": 20.38, "jug": 44.73, "istok": 20.53, "sever": 44.84}

# Zoom za ovu oblast: 14 → 64 pločice, 15 → 210 pločica.
ZOOM_MAPE = 15
MAPA_FAJL = f"beograd_osm_z{ZOOM_MAPE}.tif"   # promena zooma → novi fajl

OSM_ATRIBUCIJA = "© OpenStreetMap contributors"
RUTE_KES = "rute_kes.json"                      # već preuzete rute (štedi ORS kvotu)


NAZIVI_PROFILA = {
    "cycling-regular": "bicikl",
    "cycling-electric": "e-bicikl",
    "driving-car": "auto",
}

MAKS_POLAZAK_S = 300  # dostavljači kreću nasumično u prvih 5 minuta simulacije



# Identifikacija aplikacije prema OSM serveru pločica.
# Postavlja se kroz modul contextily.tile jer to radi u svim verzijama
# contextily-ja (parametar headers postoji tek od verzije 1.7.0).
ctx.tile.USER_AGENT = "Projekat-1a/1.0"

geo_lokacije = [
    {"naziv": "RAF", "adresa": "Kneza Mihaila 6/VI, Beograd", "lat": 44.8164, "lon": 20.4596},
    {"naziv": "ETF", "adresa": "Bulevar Kralja Aleksandra 73, Beograd", "lat": 44.8057154, "lon": 20.4762358},
    {"naziv": "DIF", "adresa": "Blagoja Parovića 156, Beograd", "lat": 44.764758, "lon": 20.428009},
    {"naziv": "Ušće Tržni Centar", "adresa": "Bulevar Mihajla Pupina 4, Beograd", "lat": 44.815549, "lon": 20.437215},
    {"naziv": "Delta City Tržni Centar", "adresa": "Jurija Gagarina 16, Beograd", "lat": 44.80541, "lon": 20.40538},
    {"naziv": "BIG Fashion Tržni Centar – Karaburma", "adresa": "Višnjička 84, Beograd", "lat": 44.81669, "lon": 20.50868},
    {"naziv": "Pravni fakultet Univerziteta u Beogradu", "adresa": "Bulevar kralja Aleksandra 67, Beograd", "lat": 44.8072, "lon": 20.4721},
    {"naziv": "Bajloni pijaca", "adresa": "Džordža Vašingtona 1B, Beograd", "lat": 44.81878, "lon": 20.46749},
    {"naziv": "Đeram pijaca", "adresa": "Živka Karabiberovića 3, Beograd", "lat": 44.80133, "lon": 20.48725},
    {"naziv": "Kalenić pijaca", "adresa": "Maksima Gorkog 2B, Beograd", "lat": 44.80014, "lon": 20.47541},
    {"naziv": "Stadion Rajko Mitić", "adresa": "Ljutice Bogdana 1A, Beograd", "lat": 44.78317, "lon": 20.46469},
    {"naziv": "Stadion Partizana", "adresa": "Humska 1, Beograd", "lat": 44.78877, "lon": 20.45907},
    {"naziv": "Farmaceutski fakultet Univerziteta u Beogradu", "adresa": "Vojvode Stepe 450, Beograd", "lat": 44.747207, "lon": 20.494823},
    {"naziv": "Cvetkova pijaca", "adresa": "Bulevar kralja Aleksandra 293, Beograd", "lat": 44.79353, "lon": 20.50559},
    {"naziv": "Pijaca Zeleni venac", "adresa": "Zeleni venac 2, Beograd", "lat": 44.81336, "lon": 20.45745},
    {"naziv": "Beogradski sajam", "adresa": "Bulevar vojvode Mišića 14, Beograd", "lat": 44.79682, "lon": 20.43553},
]

lokacija_po_nazivu = {l["naziv"]: l for l in geo_lokacije}

# ---------------------------------------------------------------------------
# Dostavljači i njihove trajektorije
# ---------------------------------------------------------------------------

def kreiraj_dostavljace(n, rng_seed=None):
    """Pravi n dostavljača sa nasumično izabranim rutama. Svaki dostavljač dobija
    drugačiji par lokacija polazak → odrediste, nasumičan profil i vreme polaska."""
    generator = random.Random(rng_seed)

    nazivi = [l["naziv"] for l in geo_lokacije]
    svi_parovi = [(od, do) for od in nazivi for do in nazivi if od != do]
    if n > len(svi_parovi):
        raise ValueError(f"Moguće je najviše {len(svi_parovi)} različitih ruta, traženo je {n}.")

    dostavljaci = []
    # sample bira n različitih parova, pa se nijedna ruta ne ponavlja
    for i, (od, do) in enumerate(generator.sample(svi_parovi, n)):
        dostavljaci.append({
            "id": i + 1,                                # jedinstveni identifikator (ključ u indeksu)
            "from": od,                                 # polazna lokacija (naziv iz geo_lokacije)
            "to": do,                                   # odredišna lokacija
            "profil": generator.choice(list(NAZIVI_PROFILA)),   # Odredjivanje da li je dostavljac biciklista, električni biciklista ili vozač automobila
            "polazak_s": generator.randrange(0, MAKS_POLAZAK_S + 1, 10), # vreme polaska u sekundama od početka simulacije (0–300 s, po 10 s)
            "boja": f"C{i % 10}",                       # boje iz matplotlib palete

            # Trajektorija: popunjava je pripremi_trajektorije(), jednom pre simulacije.
            "vremena": None,                            # vreme prolaska kroz svaku tačku rute [s]
            "lon": None,
            "lat": None,

            # Trenutna lokacija dobavljača: menja ga korak_simulacije() u svakom koraku.
            "trenutni_lon": None,
            "trenutni_lat": None,
            "status": "ceka",                           # "ceka" → "vozi" → "stigao"
        })
    
    return dostavljaci

# Odredi rutu između dve lokacije koristeći OpenRouteService API, sa keširanjem rezultata u lokalnom fajlu.
# Profil može biti "cycling-regular", "cycling-electric" ili "driving-car".
# A to znaci ili bicikl, električni bicikl ili automobil. 
# Vraća GeoJSON feature sa rutom.	
def odredi_rutu(od, do, profil):
    """Vraća rutu (GeoJSON feature) iz keša, a ako je nema tamo, iz openrouteservice-a."""
    
    kes = {}
    if os.path.exists(RUTE_KES):
        with open(RUTE_KES, encoding="utf-8") as f:
            kes = json.load(f)

    kljuc = f"{od['naziv']} -> {do['naziv']} ({profil})"
    if kljuc not in kes:
        print("Tražim rutu od Openrouting service-a:", kljuc)
        rezultat = openrouteservice_client.directions(
            coordinates=[(od["lon"], od["lat"]), (do["lon"], do["lat"])],  # (lon, lat)!
            profile=profil,
            format="geojson",
        )
        kes[kljuc] = rezultat["features"][0]
        with open(RUTE_KES, "w", encoding="utf-8") as f:
            json.dump(kes, f, ensure_ascii=False)
            
        time.sleep(1)  # Pauza od 1 sekunde zbog Rate Limit-a ORS servera

    return kes[kljuc]


def kreiraj_trajektoriju_u_vremenu(ruta, polazak_s):
    """Svakoj tački rute dodeljuje vreme prolaska (sekunde od početka simulacije).
    Pretpostavka: dostavljač vozi stalnom prosečnom brzinom cele rute."""
    
    tacke = ruta["geometry"]["coordinates"]
    trajanje_s = ruta["properties"]["summary"]["duration"]

    predjeno = [0.0]                     # pređeni put do svake tačke, u metrima
    for t1, t2 in zip(tacke, tacke[1:]):
        # ORS daje (lon, lat), a geodesic očekuje (lat, lon)!
        # Racunanje udaljenosti izmedju tacaka koristeci geopy biblioteku
        segment_distance = geodesic((t1[1], t1[0]), (t2[1], t2[0])).meters
        predjeno.append(predjeno[-1] + segment_distance)

        
    ukupno = max(predjeno[-1], 1.0)

    vremena = [polazak_s + trajanje_s * p / ukupno for p in predjeno]
    lon = [t[0] for t in tacke]
    lat = [t[1] for t in tacke]
    
    return np.array(vremena), np.array(lon), np.array(lat)


def pripremi_trajektorije_dostave(dostavljaci):
    """Za svakog dostavljača odredjuje rutu i računa trajektoriju (jednom, pre simulacije).
    Dostavljač se postavlja na početnu lokaciju."""
    
    for d in dostavljaci:
        od = lokacija_po_nazivu[d["from"]]
        do = lokacija_po_nazivu[d["to"]]
        ruta = odredi_rutu(od, do, d["profil"])

        # Odredjuje lon, lat i vreme prolaska kroz svaku tacku rute. Vremena su u sekundama od pocetka simulacije.
        d["vremena"], d["lon"], d["lat"] = kreiraj_trajektoriju_u_vremenu(ruta, d["polazak_s"])
        d["trenutni_lon"], d["trenutni_lat"] = float(d["lon"][0]), float(d["lat"][0])

        pregled = ruta["properties"]["summary"]
        print(f"Dostavljač {d['id']}: {d['from']} → {d['to']}, "
              f"{pregled['distance'] / 1000:.1f} km, {pregled['duration'] / 60:.0f} min")

# ---------------------------------------------------------------------------
# Crtanje (bez simulacije)
# ---------------------------------------------------------------------------

def iscrtaj_staticnu_sliku(dostavljaci):
    # Statični deo slike (crta se jednom): rute, polazna lokacija (■), odredišna lokacija (✖) i mapa.

    fig, ax = plt.subplots(figsize=(10, 10))    # mapa centra je skoro kvadratna

    for d in dostavljaci:
        ax.plot(d["lon"], d["lat"], color=d["boja"], linewidth=2, alpha=0.6, zorder=2,
                label=f"Dostavljač {d['id']}: {d['from']} → {d['to']} ({NAZIVI_PROFILA[d['profil']]})")
        ax.scatter(d["lon"][0], d["lat"][0], marker="s", s=60,
                   color=d["boja"], edgecolor="black", zorder=3)
        ax.scatter(d["lon"][-1], d["lat"][-1], marker="X", s=90,
                   color=d["boja"], edgecolor="black", zorder=3)

    print("Učitavam pozadinsku mapu ...")
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=MAPA_FAJL,
        reset_extent=False,
        interpolation="antialiased",
        attribution=OSM_ATRIBUCIJA,
        zorder=0,
    )

    _, _, juzna_granica, severna_granica = ax.axis()
    ax.set_aspect(1 / math.cos(math.radians((juzna_granica + severna_granica) / 2)))
    ax.set_axis_off()
    if len(dostavljaci) <= 10:          # za više dostavljača legenda bi prekrila mapu
        ax.legend(loc="lower right", fontsize=8)

    return fig, ax


def kreiraj_pokretne_elemente_na_mapi(ax, dostavljaci):
    """Pravi elemente slike koji se menjaju u svakom koraku: markere dostavljača
    (na njihovim trenutnim pozicijama) i vreme koje se ispisuje."""
    
    markeri = ax.scatter(
        [d["trenutni_lon"] for d in dostavljaci],
        [d["trenutni_lat"] for d in dostavljaci],
        # s=40, c=[d["boja"] for d in dostavljaci],
        # edgecolor="black", linewidth=1.5, zorder=5,
        s=100, c=[d["boja"] for d in dostavljaci],      # manji marker
        edgecolor="black", linewidth=1, zorder=5,       # tanji obod ostavlja više mesta za broj
    )
    # Unutar markera se ispisuje broj dostavljača (id), centrirano, belom bojom, bold fontom.
    brojevi = [
        ax.text(d["trenutni_lon"], d["trenutni_lat"], str(d["id"]),
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="white", zorder=6)
        for d in dostavljaci
    ]

    vreme_za_ispis = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=14, va="top",
                  bbox=dict(facecolor="white", alpha=0.8), zorder=6)
    
    return markeri, brojevi, vreme_za_ispis
   
def iscrtaj_korak_simulacije(markeri, brojevi, sat, pozicije, vreme):
    """Crta jedan korak: pomera markere dostavljača na nove pozicije i ažurira sat.
    Samo crta, ništa ne računa."""

    markeri.set_offsets([(p["lon"], p["lat"]) for p in pozicije])
    for broj, p in zip(brojevi, pozicije):
        broj.set_position((p["lon"], p["lat"]))

    sat.set_text(vreme.strftime("%H:%M:%S"))
    
    return (markeri, sat, *brojevi)     # blit precrtava markere, sat i sve brojeve

# ---------------------------------------------------------------------------

def lokacije_van_mape(lokacije, granice=GRANICE_BEOGRADA):
    """Vraća lokacije koje ne leže unutar granica pozadinske mape."""
    return [
        l for l in lokacije
        if not (granice["zapad"] <= l["lon"] <= granice["istok"]
                and granice["jug"] <= l["lat"] <= granice["sever"])
    ]


def dohvati_pozadinsku_mapu_beograda(granice=GRANICE_BEOGRADA, putanja_fajla=MAPA_FAJL, zoom=ZOOM_MAPE):
    """Ako nije skinuta slika mape Beograda, preuzima OSM pločice za zadate granice i snima ih kao GeoTIFF,
    ali samo ako fajl još ne postoji. 
    Ako fajl sa slikom pozadinske mape Beograda postoji, vraća putanju do fajla.
    """

    if os.path.exists(putanja_fajla):
        return putanja_fajla

    print(f"Preuzimam pozadinsku mapu BG-a (zoom {zoom}) u {putanja_fajla} ...")
    ctx.bounds2raster(
        granice["zapad"], granice["jug"], granice["istok"], granice["sever"],
        path=putanja_fajla,
        zoom=zoom,
        source=ctx.providers.OpenStreetMap.Mapnik,
        ll=True,           # granice su zadate u stepenima (lon/lat)
    )
    
    return putanja_fajla


def iscrtaj_rutu_sa_mapom_BG(putanja, mapa_fajl=MAPA_FAJL):

    longitude = [float(tacka[0]) for tacka in putanja]
    latitude = [float(tacka[1]) for tacka in putanja]

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.plot(longitude, latitude, color="royalblue", linewidth=3, zorder=3)

    ax.scatter(
        longitude[0], latitude[0],
        color="green", s=100, label=geo_lokacija_1["naziv"], zorder=4,
    )
    ax.scatter(
        longitude[-1], latitude[-1],
        color="red", s=100, label=geo_lokacija_2["naziv"], zorder=4,
    )

    # Mapa se čita iz lokalnog fajla. Fajl je u EPSG:3857, a crs="EPSG:4326"
    # kaže contextily-ju da ga prevede u lon/lat, kao i ostatak grafika.
    # reset_extent=False: prikazuje se cela mapa iz fajla, a ruta unutar nje
    # (bez toga bi se prikaz suzio na okolinu rute).
    # interpolation="antialiased": mapa se pri umanjenju ne "mreška".
    # Kod lokalnog fajla atribucija se ne dodaje sama, pa je zadajemo ručno.
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=mapa_fajl,
        reset_extent=False,
        interpolation="antialiased",
        attribution=OSM_ATRIBUCIJA,
        zorder=0,
    )

    # Korekcija odnosa osa jer stepen longitude i latitude
    # ne predstavlja istu udaljenost na geografskoj širini Beograda.
    # Računa se za sredinu cele mape, jer se ona prikazuje.
    _, _, juzna_granica, severna_granica = ax.axis()
    srednja_lat = (juzna_granica + severna_granica) / 2
    ax.set_aspect(1 / math.cos(math.radians(srednja_lat)))

    ax.set_title(f"{geo_lokacija_1['naziv']} → {geo_lokacija_2['naziv']}")
    ax.legend()
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Jednostavan primer za iscrtavanje rute između dve lokacije u Beogradu, sa pozadinskom mapom.
    # Ne koristi se u glavnoj simulaciji, ali može biti koristan za testiranje i demonstraciju.
    
    for l in lokacije_van_mape(geo_lokacije):
        print(f"UPOZORENJE: {l['naziv']} ({l['lat']}, {l['lon']}) je van pozadinske mape.")

    dohvati_pozadinsku_mapu_beograda()

    geo_lokacija_1 = geo_lokacije[0]    # RAF
    geo_lokacija_2 = geo_lokacije[10]   # Stadion Crvene Zvezde

    # Redosled koordinata je (longitude, latitude)!
    lokacije_rute = [
        (geo_lokacija_1["lon"], geo_lokacija_1["lat"]),
        (geo_lokacija_2["lon"], geo_lokacija_2["lat"])
    ]

    # Odredi rutu između dve lokacije koristeći OpenRouteService API.
    rezultat = openrouteservice_client.directions(
        coordinates=lokacije_rute,
        profile="cycling-regular",   # Za automobil: "driving-car"
        format="geojson",
    )

    ruta = rezultat["features"][0]
    putanja = ruta["geometry"]["coordinates"]
    pregled = ruta["properties"]["summary"]

    print(f"Ruta od {geo_lokacija_1['naziv']} do {geo_lokacija_2['naziv']}:")

    print("Dužina puta:", pregled["distance"], "m")
    print("Procenjeno trajanje:", pregled["duration"], "s")
    print("Broj tačaka putanje:", len(putanja))
    print("Prvih pet tačaka putanje:", putanja[:5])

    iscrtaj_rutu_sa_mapom_BG(putanja=putanja)
