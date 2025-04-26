# Automatic recognition of benthic meiofauna using machine learning / Meiobentose hulka kuuluvate organismide automaatne tuvastamine masinõppe meetoditega 

Selles repositooriumis on bakalaureusetöö raames valminud ja meiobentost fotodelt tuvastavate mudelite treenimiseks kasutatud programmide kood. Programmide koostamisel kasutati ChatGPT abi. Sisend ja ChatGPT vastused on lisatud siia repositooriumisse.

Meiobentose tuvastamiseks sobivate masinõppemeetodite hindamiseks koostati mikrofotodest andmestik, mis jagati märgendatud treening- ja valideerimisandmeteks ning märgendamata testandmeteks. Kirjanduse ja andmete iseloomu põhjal valiti välja kaks masinõppemeetodit: Faster R-CNN ja YOLO. Nende masinõppemeetoditega treeniti treeningandmetel mudelid, mis tuvastavad meiobentose organismid ja identifitseerivad millisesse rühma organism kuulub. 

Märgendatud treening- ja valideerimisandmed on ... . Siin repositooriumis on näidiseks vaid esimesed kümme fotot nii treening- kui valideerimisandmetest. 

# Faster R-CNN
Mudeli treenimiseks on 2 skripti: 
* benthos_train.py - kasutab märgendatud andmestikku, mis sisaldab 9 objektiklassi (nematode + rotifer + Testacea + ciliate + turbellarians + annelida + arthropoda + gastrotricha + tardigrada). Sama programmi saab kasutada ka vähemate objektiklassidega mudeli treenimiseks, kui haruldasem(ad) rühm(ad) eemaldada (ka treeningandmetest)
* benthos_train_detect_animal_only.py - treenib mudeli(d), mille eesmärgiks on vaid tuvastada organismid (mistahes meiobentose esindajad)

  Treenitud mudelite hindamiseks on samuti 2 skripti:
* benthos_test.py - 9 klassi eraldi tuvastamiseks treenitud mudelite hindamiseks: precision, recall, F1-score, accuracy, mAP50, mAP50-95
* benthos_test_detect_animal_only.py - vaid meiobentose esindajate tuvastamiseks treenitud mudelite hindamiseks: precision, recall, F1-score, accuracy, mAP50, mAP50-95

  visualize_prediction.py - võtab testandmete kaustast pildid ja tuvastab neil esinevad organismid valitud mudelitega. Märgib pildile leitud organismid koos tõenäosustega, salvestab tuvastusruutudega pildid tulemuste kausta(desse) ja genereerib tulemuste raporti.

  annotation_file_extractor.py - märgenduste failist võtab iga pildi kohta sellel esinevad märgendused (klasside nimed) ja teeb ülevaatliku exceli faili, kust saab hõlpsalt kindlaks teha mis organismid millistel piltidel esinesid ja kokku lugeda palju iga klassi esindajaid on pildimaterjalil kokku.

# YOLO

 
