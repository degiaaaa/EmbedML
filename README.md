# 1. Maschinelles Lernen

Maschinelles Lernen (Machine Learning, ML) ist ein bedeutender Bereich der künstlichen Intelligenz, der Computern ermöglicht, aus Daten zu lernen und Muster zu erkennen, ohne dass eine explizite Programmierung erforderlich ist. In diesem Abschnitt werden die Grundlagen des Maschinellen Lernens behandelt. Der Fokus liegt dabei auf der Auswahl der Themen, die gezielt auf die Anwendung von TinyML (Kapitel 2) ausgerichtet sind. [Quelle/weitere Infos](https://www.ibm.com/de-de/topics/machine-learning)

Maschinelles Lernen (ML) ist ein Teilgebiet der Künstlichen Intelligenz (KI), das Computern ermöglicht, aus Daten zu lernen und Entscheidungen zu treffen, ohne explizit programmiert zu sein. Es basiert auf Algorithmen, die Daten analysieren und Muster erkennen, um Vorhersagen oder Entscheidungen zu treffen. KI umfasst ein breiteres Spektrum an Technologien, einschließlich maschinelles Lernen, bei dem Maschinen Aufgaben ausführen, die menschliche Intelligenz erfordern, wie z.B. Spracherkennung, Problemlösung und Lernen. Deep Learning ist eine spezielle Klasse von maschinellen Lernverfahren, die auf künstlichen neuronalen Netzen basieren. Diese Netze sind in der Lage, komplexe Muster in großen Datenmengen zu erkennen. Während maschinelles Lernen also die Techniken und Methoden umfasst, durch die Computer aus Daten lernen können, spezialisiert sich Deep Learning auf tiefere, komplexere Strukturen, die das Lernen noch weitergehender und feingranularer ermöglichen.

<figure><img src=".gitbook/assets/Maschinelles_Lernen (2).svg" alt=""><figcaption><p>Abbildung 1: Teilbereiche künstlicher Intelligenz</p></figcaption></figure>

**1.1 Grundlagen des maschinellen Lernens**

Im Bereich des maschinellen Lernens und des TinyML ist es von grundlegender Bedeutung, die verschiedenen Arten des Maschinellen Lernens und den dazugehörigen Workflow zu verstehen. TinyML bezeichnet die Implementierung von Machine Learning auf extrem ressourcenbeschränkten Geräten, wie Mikrocontrollern und IoT-Geräten. In diesem Abschnitt wird auf die Grundlagen des maschinellen Lernens und dessen Konzepte, die für die Anwendung von TinyML bedeutend sind, eingegangen.

#### 1.1.1 **Arten des Maschinellen Lernens**

Es gibt verschiedene Formen des Maschinellen Lernens, die sich in ihrem Ansatz und ihren Anwendungen unterscheiden. Die Hauptkategorien hierbei sind:

*   **Überwachtes Lernen (Supervised Learning)**

    Beim überwachten Lernen werden Modelle anhand von Eingabe- und Ausgabedaten trainiert. Das bedeutet, dass das Modell während des Trainings bereits mit den richtigen Antworten versorgt wird. Diese Art des Lernens eignet sich für Aufgaben wie Klassifikation und Regression. [IBM/Was ist überwachtes Lernen?](https://www.ibm.com/de-de/topics/supervised-learning)
*   **Halb-Überwachtes Lernen (Semi-Supervised Learning)**

    Beim halb-überwachten Lernen wird das Modell mit einer Kombination aus gelabelten und ungelabelten Daten trainiert. Diese Methode kann besonders nützlich sein, wenn das Labeln von Daten zeitaufwändig oder teuer ist, da sie die Vorteile des überwachten Lernens mit der Effizienz des unüberwachten Lernens kombiniert. [Quelle/weitere Infos](https://www.ibm.com/topics/semi-supervised-learning)
*   **Unüberwachtes Lernen (Unsupervised Learning)**

    Im unüberwachten Lernen gibt es keine Ausgabedaten. Im Training lernt das Modell Muster und Strukturen in den Eingabedaten zu erkennen. Zu den Aufgaben des unüberwachten Lernens gehören Clustering und Dimensionsreduktion. [Quelle/weitere Infos](https://www.ibm.com/de-de/topics/unsupervised-learning)
*   **Bestärkendes Lernen (Reinforcement Learning)**

    Beim bestärkenden \*\*\*\*Lernen lernt ein Modell durch Interaktion mit einer Umgebung. Während der Lernphase des Modells trifft es Entscheidungen und erhält Belohnungen oder Bestrafungen als Rückmeldung. Diese Art des Lernens findet Anwendung in Bereichen wie Robotik und Spielstrategien. [Quelle/weitere Infos](https://datasolut.com/reinforcement-learning/)

**Unterschiede zwischen den Arten des Maschinellen Lernens**

Die verschiedenen Formen des Maschinellen Lernens unterscheiden sich in ihrem Ansatz und den verwendeten Daten. Überwachtes Lernen nutzt Eingabe- und Ausgabedaten, unüberwachtes Lernen erkennt Muster ohne Ausgabedaten. Halb-überwachtes Lernen kombiniert gelabelte und ungelabelte Daten, während bestärkendes Lernen auf Interaktion und Rückmeldungen basiert.

#### 1.1.2 Maschinelles Lernen - Workflow

Der Workflow des Maschinellen Lernens umfasst die folgenden Schritte:

1. **Datenvorverarbeitung:** _Beschaffung:_ Daten sammeln aus verschiedenen Quellen. _Bereinigung:_ Entfernen von Fehlern, Auffüllen fehlender Werte, Anpassen von Datenformaten. _Aufteilung:_ Daten in Trainings-, Validierungs- und Testdatensätze aufteilen für das Modelltraining und die Leistungsbeurteilung.
2. **Modellauswahl**: _Problemdefinition:_ Klarstellung des Problems, ob es sich um ein Supervised- oder Unsupervised-Learning handelt. _Algorithmenwahl:_ Auswahl eines geeigneten Modells basierend auf der Aufgabenstellung, z.B., Klassifikation (Logistische Regression), Clustering (K-Means) usw. _Modelltyp:_ Entscheidung zwischen eigenem Modelltraining oder Verwendung eines vortrainierten Modells, je nach Datenmenge und spezifischen Anforderungen.
3. **Modelltraining**: Das Modell wird mit den Trainingsdaten trainiert und auf dem Validierungsdatensatz validiert, um Muster zu erkennen und Vorhersagen zu treffen.
4. **Modellevaluation**: Das Modell wird auf dem Testdatensatz getestet, um seine Leistung zu bewerten.
5. **Monitoring**: Nach der Bereitstellung des Modells ist es wichtig, das Modell kontinuierlich zu überwachen. Dies beinhaltet die Überwachung seiner Leistung im Einsatz und gegebenenfalls die Aktualisierung des Modells, um sicherzustellen, dass es weiterhin genaue Vorhersagen trifft.

Dies sind die grundlegenden Schritte und Konzepte im maschinellen Lernprozess. In den kommenden Kapiteln wird eingehender auf verschiedene Algorithmen und Techniken dieser Bereiche eingegangen, Siehe Kapitel 2.5 Neural Networks and Deep Learning.

[GoogleCloud/ML-Workflow](https://cloud.google.com/ai-platform/docs/ml-solutions-overview?hl=de)

#### **1.1.3 Frameworks**

Frameworks sind essentielle Werkzeuge im Maschinellen Lernen, die Entwicklern die Implementierung von Modellen und Algorithmen erleichtern. Hier sind einige der wichtigsten Frameworks, die in der ML-Community verwendet werden:

*   **TensorFlow**

    TensorFlow ist ein Open-Source-Framework, das von Google entwickelt wurde. Es ist besonders bekannt für seine Flexibilität und Skalierbarkeit. TensorFlow ermöglicht das Erstellen und Trainieren von neuronalen Netzen, sowie die Implementierung von vortrainierten Machine-Learning-Modellen. Keras ist ein benutzerfreundliches Deep Learning-Framework, das in TensorFlow integriert ist und die Entwicklung von Modellen erleichtert.

    TensorFlow Lite ist eine spezielle Version von TensorFlow, die für die Ausführung von maschinellen Lernmodellen auf eingebetteten Geräten und Mikrocontrollern optimiert ist, wodurch die Implementierung von Modellen auf ressourcenbeschränkten Edge-Geräten ermöglicht wird. [Quelle/weitere Infos](https://www.databricks.com/de/glossary/tensorflow-guide)
*   **PyTorch**

    PyTorch ist ein weiteres beliebtes Open-Source-Framework, das von Facebook AI entwickelt wurde. Es zeichnet sich durch seine dynamische Berechnungsgraphen aus, was es ideal für Forschungszwecke macht. PyTorch bietet eine einfache und intuitive Schnittstelle, um neuronale Netzwerke zu entwickeln und zu trainieren. [Quelle/weitere Infos](https://www.ibm.com/de-de/topics/pytorch)
*   **scikit-learn und xgboost**

    scikit-learn ist eine weit verbreitete Bibliothek für Machine Learning in Python. Es bietet eine Vielzahl von Werkzeugen für Datenaufbereitung, Modellauswahl und Modellbewertung. xgboost, kurz für "Extreme Gradient Boosting", ist eine effiziente Implementierung von Gradient Boosting-Verfahren. Diese Bibliotheken werden meist für klassische Ansätze von Machine Learning eingesetzt wie zum Beispiel Random Forest. [Quelle/weitere Infos](https://databasecamp.de/python/scikit-learn), [Quelle/weitere Infos 2](https://databasecamp.de/ki/xgboost)

Diese Frameworks sind eine ausgezeichnete Grundlage für die Entwicklung von Machine-Learning-Anwendungen. In den folgenden Kapiteln werden die Frameworks TensorFlow und scikit-learn für Beispiele verwenden.
