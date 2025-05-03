# Automatic recognition of benthic meiofauna using machine learning / Meiobentose hulka kuuluvate organismide automaatne tuvastamine masinõppe meetoditega 

Selles repositooriumis on bakalaureusetöö raames valminud ja meiobentost fotodelt tuvastavate mudelite treenimiseks kasutatud programmide kood. Programmide koostamisel kasutati ChatGPT abi. Sisend ja ChatGPT vastused on lisatud siia repositooriumisse: fail nimega ChatGPT vestlused.pdf.

Meiobentose tuvastamiseks sobivate masinõppemeetodite hindamiseks koostati mikrofotodest andmestik, mis jagati märgendatud treening- ja valideerimisandmeteks ning märgendamata testandmeteks. Kirjanduse ja andmete iseloomu põhjal valiti välja kaks masinõppemeetodit: Faster R-CNN ja YOLO. Nende masinõppemeetoditega treeniti treeningandmetel mudelid, mis tuvastavad meiobentose organismid ja identifitseerivad millisesse rühma organism kuulub. 

Märgendatud treening- ja valideerimisandmed on kättesaadavad Kaggle platformil: (https://www.kaggle.com/datasets/kyllilokko/meiobenthos-dataset). Siin repositooriumis on näidiseks vaid esimesed kümme fotot nii treening- kui valideerimisandmetest. 

# Faster R-CNN
Mudeli treenimiseks on 3 skripti: 
* train_models_8groups.py - kasutab märgendatud andmestikku, mis sisaldab 8 objektiklassi (nematode + rotifer + Testacea + ciliate + turbellarians + annelida + arthropoda +  tardigrada). 
* train_models_9groups.py - kasutab märgendatud andmestikku, mis sisaldab 9 objektiklassi (nematode + rotifer + Testacea + ciliate + turbellarians + annelida + arthropoda + gastrotricha + tardigrada). 
* train_models_detect_animal_only.py - treenib mudeli(d), mille eesmärgiks on vaid tuvastada organismid (mistahes meiobentose esindajad)

Treenitud mudelite hindamiseks on samuti 3 skripti:
* validate_models_8groups.py - 8 klassi eraldi tuvastamiseks treenitud mudelite hindamiseks: precision, recall, F1-score, accuracy, mAP50, mAP50-95
* validate_models_9groups.py - 9 klassi eraldi tuvastamiseks treenitud mudelite hindamiseks: precision, recall, F1-score, accuracy, mAP50, mAP50-95
* validate_models_detect_animal_only.py - vaid meiobentose esindajate tuvastamiseks treenitud mudelite hindamiseks: precision, recall, F1-score, accuracy, mAP50, mAP50-95
* need 3 ülaltoodud skripti kasutavad skripte coco_eval.py ja coco_utils.py

visualize_prediction.py - võtab testandmete kaustast pildid ja tuvastab neil esinevad organismid valitud mudelitega. Märgib pildile leitud organismid koos tõenäosustega, salvestab tuvastusruutudega pildid tulemuste kausta(desse) ja genereerib tulemuste raporti.

# YOLO
COCO_to_YOLO_conversion.py - konverteerib COCO formaadis märgendused YOLO jaoks sobivasse formaati
  
Mudeli treenimiseks ja treenitud mudelite hindamiseks on skript: 
* train_YOLO11_models.py - treenib ja kohe ka hindab treenitud mudelid ning salvestab logi. Vajalik kontrollida ja vajadusel muuta märgendatud andmestiku asukoha, data.yaml faili sisu (märgendite ja andmete kaustade asukohad, objektiklasside arv ja nimetused), treenitud mudelite ja logi salvestuskoht.

visualize_prediction.py - võtab testandmete kaustast pildid ja tuvastab neil esinevad organismid valitud mudelitega. Märgib pildile leitud organismid koos tõenäosustega, salvestab tuvastusruutudega pildid tulemuste kausta(desse) ja genereerib tulemuste raporti.

  
 
