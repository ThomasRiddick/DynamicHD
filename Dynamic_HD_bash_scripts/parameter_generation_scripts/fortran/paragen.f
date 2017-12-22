      PROGRAM PARAGEN
c
c     ******** Programm zur Generierung der Parameter fuer Overland- und
c              Riverflow. Da hierfuer jeweils die lineare Speicherkaskade
c              vorgesehen ist, werden die Koeffizienten n und k generiert.
c
c     ******** Fuer einen ersten Schnellschuss werden die Parameter n, k
c              mit den Sausenkoeffizienten korreliert unter der Analogie-
c              Annahme, dass sich die Koeffizienten global aehnlich
c              zueeinander verhalten wie in den Vindelaelven-Catchments.
c              Wobei hier die optimierten Sausen-Koeffizienten als
c              Grundlage genommen werden.
c
c              Als weitere erste Annahme wird die Speicherzahl n als
c              konstant angesetzt. Fuer See- und Gletscherpunkte werden
c              die Parameter nicht definiert, d.h. sie sind 0. Gletscher
c              sind vorhanden in der Antarktis und auf Groenland.
c
c              Als Datenformate fuer Ein- und Auslese sind sowohl
c              das Cray-Binaerformat als auch das Globale Binaerformat
c              vorgesehen. Es ist eine Input-Datei PARAGEN.INP von Noeten,
c              in der verschiedene Maschinen-spezifische Eingabeparameter
c              gesetzt werden.
c
c              Auf der Cray ist die Basis-LU = 50 ==> Assign unblocked noetig
c              fuer 50
c
c     ******** Programmierung und Entwicklung: Stefan Hagemann
c              Version 1.0 -- September 1995
c
c     ******** Version 1.1 -- Januar 1996
c              Einbau eines konstanten Modifizierungsfaktor fuer die k-Werte
c              und/oder n-Werte ==> aus Initialisierungsdatei eingelesen
c
c     ******** Version 1.2 -- Februar 1996
c              Einbau des erstellens der Baseflow-k-Werte und der
c              Initialisierungsdatei fuer den linearen Baseflowspeicher
c              --> Nur falls gewuenscht.
c
c     ******** Version 1.3 -- Sicherheitskorrektur Oktober 1996
c
c     ******** Version 1.4 -- November-Dezember 1996
c              New Methods of Parametrization Approaches
c              Ausgabe der Slope-Datei = dh/dx
c
c     ******** Version 1.5 -- Januar 1997
c              New Methods of Parametrization Approaches, u.a. Heavyside-Ansaetze
c
c     ******** Version 1.6 -- Februar 1997
c              Einbau von Lakes und Wetlands sowie der Baseflowparameterisierung
c
c     ******** Linkroutinen:  PARAGEN.for
c                             GLOBUSE.for
c                             MATHE.for
c                             MODTIME.for
c
c     ******** Variablenliste
c     ***
c     ***     NL = Anzahl der Laengenkreise (Longitudinal-Gitterboxen)
c     ***     NB = Anzahl der Breitenkreise (Latitudinal-Gitterboxen)
c     ***   AREA = Gitterbox-Flaechenarray [m^2]
c     ***   DLAT = Breitenabstand (konstant)[m]
c     ***   DLON = Array der Laengenabstaende [m]
c     ***   FORO = Orographiearray [m]
c     ***   TSIG = Array der Streuung der Orographie [m]
c     ***   FDIR = Riverdirectionarray
c     ***  ALF_K = Array der Retentionskonstanten k - Overlandflow [day]
c     ***  ALF_N = Array der Speicherzahlen n       - Overlandflow
c     ***  ARF_K = Array der Retentionskonstanten k - Riverflow [day]
c     ***  ARF_N = Array der Speicherzahlen n - Riverflow
c     ***  AGF_K = Array der Retentionskonstanten k - Baseflow [day]
c     ***   FGSP = Initialisierungsarray fuer den linearen Baseflowspeicher
c     ***   FLAG = Landmasken-Array (= Landseemaske ohne Lakes)
c     ***  GFLAG = Gletschermaske
c     ***  SLOPE = Slope = dh/dx
c     ***  SLINN = Inner Slope Array
c     ***          Wird in Part 2 als Wetland Percentage Array benutzt [%]
c     ***    NIS = Anzahl der Landgitterboxen mit SLINN = 0
c     ***    FDD = Drainage Density [%/km ]
c     ***          Wird in Part 2 als Lake Percentage Array benutzt [%]
c     ***
c     ***  FLVEL = Velocity-Array fuer Overlandflow [m/s]
c     ***  FRVEL = Velocity-Array fuer Riverflow [m/s]
c     ***
c     ***
c     ***  IFIN/IFOUT = Ein- bzw. Ausgabe-Format
c     ***          1 = Cray-Binaerformat
c     ***          2 = REGEN: Globales Binaerformat
c     ***          3 = REGEN: Waveiso2-Format
c     ***     LU = Logical Unit (50 fuer Cray, 20 fuer REGEN)
c     ***    LUF = Logical Unit fuer Flaechen-und Abstandsfile = 30
c     ***
c     ***  DNINP = Inputdateinamen, z.B. fuers globale Orographie-Array,
c     ***          Landmaske, Gletschermaske, globale Flaechen/Abstandsarray
c     ***          Riverdirectionfiles, Drainage Density Array
c     ***  DNOUT = Ausgabedatei (over_k.dat, over_n.dat, riv_k.dat, riv_n.dat)
c     ***
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c     ***  IPARA = Art der Parameterisierung
c     ***          1 = Analog zu den Sausen-Koeffizienten mit reell
c     ***          2 = Analog zu den Sausen-Koeffizienten mit n integer und
c     ***              n_reell * k = n_integer * k_mod = Lag konst.
c     ***          3 = 1 & lineare Abhaengigkeit von Topography-Streuung
c     ***          4 = 1 & bolische Abhaengigkeit (O.) von Topography-Streuung
c     ***          5 = 1 & Bolische Abhaengigkeit (R.) von Topography-Streuung
c     ***          6 = 1 & Quadratische Abh. (O.) von der Topographie-Streuung
c     ***          7 = 1 & Heavyside-Abh. (O.) von Topography-Streuung
c     ***          8 = 1 & Inner Slope statt Slope f. Overlandflow
c     ***
c     ***         11 = Singh or other
c     ***
c     ***  FK_LFK = Modifizierungsfaktor fuer k-Werte beim Overlandflow
c     ***  FK_LFN = Modifizierungsfaktor fuer n-Werte beim Overlandflow
c     ***  FK_RFK = Modifizierungsfaktor fuer k-Werte beim Riverflow
c     ***  FK_RFN = Modifizierungsfaktor fuer n-Werte beim Riverflow
c     ***  FK_GFK = Modifizierungsfaktor fuer k-Werte beim Baseflow
c     ***
c     ***  IBASE = Modell der Baseflowparameterisierung
c     ***           0 =   k = 300 days
c     ***           1 =   k = DX / 50 km * 300 days
c     ***           2 =   k = 300 days / Orographiefaktor
c     ***           3 =   k = DX / 50 km * 300 days / Orographiefaktor
c     ***           4 =   new ideas
c     ***
c     ***  ILAMOD = Modell der Lake-Dependance
c     ***           0 = No Lake-dep.
c     ***           1 = Charbonneau-Ansatz
c     ***           2 = tanh-Ansatz
c     ***           3 = linearer Ansatz
c     ***  VLA100 = Flow-Velocity bei 100 % Lake-Percentage [m/s]
c     ***  ISWMOD = Modell der Wetland-Dependance
c     ***           0 = No Swamp-dep.
c     ***           1 = Swamps werden den Lakes hinzuaddiert
c     ***           2 = tanh-Ansatz
c     ***           3 = linearer Ansatz
c     ***           4 = tanh nur fuer Overlandflow
c     ***           5 = tanh mit Wetlandtypes bei Riverflow
c     ***           6 = tanh mit Permafrost bei Riverflow
c     ***  VSW100 = Flow-Velocity bei 100 % Swamp-Percentage [m/s]
c     ***  PROARE = Area-Percentage, ab die Lake/Swamp-Percentage sich auswirkt
c     ***    VSAU = Minimum-Sausen-Velocity = principal dummy
c     ***
c
      PARAMETER (NL = 720, NB = 360)
      PARAMETER (C = 2., ALPHA = 0.1, PI = 3.1415927)
c
      REAL AREA(NB), DLAT, DLON(NB), FORO(NL, NB), FDIR(NL, NB)
      REAL ALF_K(NL, NB), ALF_N(NL, NB)
      REAL ARF_K(NL, NB), ARF_N(NL, NB)
      REAL AGF_K(NL, NB)
c###     REAL FGSP(NL, NB)
      REAL SLOPE(NL, NB), SLINN(NL, NB)
      REAL FLAG(NL, NB), GFLAG(NL, NB), TSIG(NL, NB)
      REAL FLVEL(NL, NB), FRVEL(NL, NB), FDD(NL, NB)
      REAL PROARE, VLA100, VSW100, XIB, BB
      CHARACTER*1024 DNINP, DNOUT,ZEILE
      CHARACTER CINI*6
      INTEGER IFIN, IFOUT, IPARA, IQUE, NDUM, IGMEM, ILAMOD
      INTEGER ISWMOD, IBASE
      REAL FDUM, FK_LFK, FK_LFN, FK_RFK, FK_RFN, FK_GFK, TSIG0
c
c     *** Vorbelegungen
      LUF = 30
c
c     *** Nullinitialisierungen der Felder
      DO JB=1, NB
      DO JL=1, NL
         ALF_K(JL, JB) = 0.
         ALF_N(JL, JB) = 0.
         ARF_K(JL, JB) = 0.
         ARF_N(JL, JB) = 0.
         AGF_K(JL, JB) = 0.
c###         FGSP(JL, JB) = 0.
         SLOPE(JL, JB) = 0.
      ENDDO
      ENDDO
c
c     ******* externe Belegungen aus Inputdatei PARAGEN.inp
      CINI = "IFIN"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IFIN = INT(FDUM+0.01)
      CINI = "IFOUT"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IFOUT = INT(FDUM+0.01)
      CINI = "IPARA"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IPARA = INT(FDUM+0.01)
      CINI = "IQUE"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IQUE = INT(FDUM+0.01)
c
c     *** Modifizierungsfaktoren der Parameter
      CINI = "FK_LFK"
      CALL PARINP(LUF, CINI, FK_LFK, ZEILE, IQUE)
      CINI = "FK_LFN"
      CALL PARINP(LUF, CINI, FK_LFN, ZEILE, IQUE)
      CINI = "FK_RFK"
      CALL PARINP(LUF, CINI, FK_RFK, ZEILE, IQUE)
      CINI = "FK_RFN"
      CALL PARINP(LUF, CINI, FK_RFN, ZEILE, IQUE)
      CINI = "FK_GFK"
      CALL PARINP(LUF, CINI, FK_GFK, ZEILE, IQUE)
c
c     *** Cray oder REGEN ?
      IF (IFIN.EQ.1) THEN
         LU=50
      ELSE
         LU=20
      ENDIF
c
c     ******* Input-Dateien-Auslese
c
c     *** Orographie
      CINI = "TDNORO"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, FORO, NL, NB, IQUE)
c
c     *** Orographie-Streuung
      CINI = "TDNSIG"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, TSIG, NL, NB, IQUE)
c
c     *** Landmaske
      CINI = "TDNMAS"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, FLAG, NL, NB, IQUE)
c
c     *** Gletschermaske
      CINI = "TDNGMA"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, GFLAG, NL, NB, IQUE)
c
c     *** Globaler Flaechen/Abstandsfile
      CINI = "TDNFL"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL AREAREAD(LUF, DNINP, AREA, DLON, DLAT, NB, IQUE)
c
c     *** Riverdirection-File
      CINI = "TDNDIR"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, FDIR, NL, NB, IQUE)
c
c     *** Drainage Density
      CINI = "TDNDD"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, FDD, NL, NB, IQUE)
c
c     *** Inner Slope
      CINI = "TDNSLI"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, SLINN, NL, NB, IQUE)
c
c     ********* Weiterverarbeitung der Inputarrays
C
C     *** Orographie nur auf Landpunkten
      DO JB = 1, NB
      DO JL = 1, NL
         FORO(JL, JB) = FORO(JL,JB) * FLAG(JL, JB)
c         IF (FORO(JL, JB).LT.0.0) THEN
c            WRITE(*,*) JL, JB, FORO(JL, JB)
c            FORO(JL, JB)=0.0
c         ENDIF
c
c        *** Minimum-Orographiestreuung = 0.1
         IF (FLAG(JL, JB).GT.0.5 .AND. GFLAG(JL, JB).LT.0.5
     &         .AND. TSIG(JL, JB).LE.0.0) THEN
           IF (IQUE.NE.0) WRITE(*,*) 'Streuung: ',JL,JB,TSIG(JL, JB)
           TSIG(JL, JB)=0.1
         ENDIF

      ENDDO
      ENDDO
      WRITE(*,*) 'Orography und Streuung ueberprueft'
c
c     ******* Basiswerte bzgl. Vindelaelven-Catchments:
c
      ALF_K0 = 16.8522
      ALF_N0 = 2.2214
ccc      ALF_V0 = 0.0588
      ALF_V0 = 1.0885
      ALF_DX = 171000.
c
      ARF_K0 = 0.4112
      ARF_N0 = 9.1312
ccc      ARF_V0 = 0.385
      ARF_V0 = 1.0039
      ARF_DX = 228000.
c
c     *** From Torneaelven: weighted mean der Topographiestreuung
      TSIG0 = 64.8669
c
c     *** IPARA=4
      IF (IPARA.EQ.4) THEN
         X0 = 10
         X1 = TSIG0
         X2 = 200
         Y0 = 4
         Y1 = 1
         Y2 = 1./Y0
c
c        *** Idee: Y2 = 1 / Y0 = 0.5
         IQUE=2
         CALL BOLIC(X0,Y0, X1,Y1, X2,Y2, QA,QB,QC, IQUE)
         WRITE(*,*) " QA = ", QA
         WRITE(*,*) " QB = ", QB
         WRITE(*,*) " QC = ", QC
         IF (IQUE.NE.0) GOTO 999
c
c     *** IPARA=5
      ELSE IF (IPARA.EQ.5) THEN
         X0 = 20
ccc         X1 = TSIG0
         X1 = 50
         X2 = 200
         Y0 = 4
         Y1 = 1
         Y2 = 0.8
c
c        *** Idee: Y2 = 1 / Y0 = 0.5
         IQUE=2
         CALL BOLIC(X0,Y0, X1,Y1, X2,Y2, QA,QB,QC, IQUE)
         WRITE(*,*) " QA = ", QA
         WRITE(*,*) " QB = ", QB
         WRITE(*,*) " QC = ", QC
         IF (IQUE.NE.0) GOTO 999
      ENDIF
c
c     ******* Verzweigung nach Art der Parameterbestimmung
      IF (IPARA.LT.1 .OR. IPARA.GT.11) GOTO 999
c
c     ******* Parameterberechnung *****************************************
c
      NIS = 0
      DO 150 JB = 1, NB
      DO 150 JL = 1, NL
         IF (FLAG(JL, JB).LT.0.5 .OR. GFLAG(JL, JB).GT.0.5) GOTO 150
c
c        *** IL, IB = relative Richtungskoordinaten
c        *** Die 0.1-Summanden sind noetig wegen Cray-Rundungsungenauigkeiten
         IB = -( INT( (FDIR(JL,JB)-1.)/3. + 0.1) - 1)
         IL = INT( ( (FDIR(JL,JB)+2.)/3. -
     &        INT((FDIR(JL,JB)+2.)/3. + 0.1) ) * 3. + 0.1) -1
c
c        *** Lokale Senke ?
         IF (IL.EQ.0 .AND. IB.EQ.0) GOTO 150
C
         JLNEW = JL + IL
         JBNEW = JB + IB
c
c        *** Greenwichgrenze
         IF (JLNEW.EQ.0) JLNEW = NL
         IF (JLNEW.EQ.NL+1) JLNEW = 1
c
         DX = SQRT(IB*IB*DLAT*DLAT + IL*IL*DLON(JB)*DLON(JB))
         DH = FORO(JL, JB) - FORO(JLNEW, JBNEW)
         SLOPE(JL, JB) = DH / DX
c
c        *** Kommentar ?
         IF (IQUE.NE.0 .AND. JB.GE.40 .AND. JB.LE.42 .AND.
     &       JL.GE.394 .AND. JL.LE.410) THEN
            WRITE(*,*) "JL = ", JL, "  JB =", JB, "  FDIR = ",
     &             FDIR(JL, JB), " ==> IL = ", IL, "  IB = ", IB
            WRITE(*,*) "DX = ", DX, "  und DH =", DH
         ENDIF
c
c        *** Minimales V, das entspricht Minimum-Steigung
c        *** Ist nur Dummy, da DH = 0 nur bei lokalen Senken vorkommen darf.
c        Minimum is now set to (VSAU/C)**(1/ALPHA)*DX
c        which is (0.1/2)**(1/0.1)*50000
         IF (DH .LE. 0) THEN
            VSAU = 0.65
         ELSE IF (DH.LE. 0.00000000488281 ) THEN
c
ccc            VSAU = 0.01   (0.1 = 5.79 days auf 50 km)
            VSAU = 0.1
            WRITE(*,*) "Achtung: JL = ",JL, "  JB =",JB, "  FDIR = "
     &         ,  FDIR(JL, JB), "DX = ", DX, "  und DH =", DH
         ELSE
            VSAU = C * (DH/DX)**ALPHA
         ENDIF

c
c        ******** Sausen -Analogie
         IF (IPARA.LE.7) THEN
c
c           *** Overlandflow
            ALF_K(JL, JB) = ALF_K0 * DX/ALF_DX * ALF_V0/VSAU
            ALF_N(JL, JB) = ALF_N0
c
c           *** Riverflow
            ARF_K(JL, JB) = ARF_K0 * DX/ARF_DX * ARF_V0/VSAU
            ARF_N(JL, JB) = ARF_N0
c
c        *** Sausen & Inner Slope statt Slope f. Overlandflow
         ELSE IF (IPARA.EQ.8) THEN
c
c           *** Overlandflow
            DXO = SQRT(DLAT*DLAT + DLON(JB)*DLON(JB))
            IF (SLINN(JL, JB).GT.0) THEN
               VSO = C * SLINN(JL, JB)**ALPHA
               ALF_K(JL, JB) = ALF_K0 * DXO/ALF_DX * ALF_V0/VSO
            ELSE
c
c              *** Minimales V, das entspricht Minimum-Steigung
c              ***    ==> Inner Slope durch Normal Slope ersetzen
c
ccc            WRITE(*,*) "Achtung: JL = ",JL, "  JB =",JB, "  FDIR = "
ccc     &         ,  SLINN(JL, JB)
               ALF_K(JL, JB) = ALF_K0 * DX/ALF_DX * ALF_V0/VSAU
               NIS = NIS + 1
            ENDIF
            ALF_N(JL, JB) = ALF_N0
c
c           *** Riverflow
            ARF_K(JL, JB) = ARF_K0 * DX/ARF_DX * ARF_V0/VSAU
            ARF_N(JL, JB) = ARF_N0
c
c           *** Anwendung der Multiplikationsfaktoren aus Torneaelven-Experimenten
            ALF_K(JL,JB) = ALF_K(JL,JB) * 3.
            ALF_N(JL,JB) = ALF_N(JL,JB) * 0.5
            IF (ALF_N(JL,JB).LE.0.5) THEN
              NDUM = INT(ALF_N(JL,JB)+ 0.5)
              ALF_K(JL,JB) = ALF_K(JL,JB)*ALF_N(JL,JB)/NDUM
              ALF_N(JL,JB) = NDUM
            ENDIF
c
            ARF_N(JL,JB) = ARF_N(JL,JB) * 0.6
            IF (ARF_N(JL,JB).LE.0.5) THEN
               NDUM = INT(ARF_N(JL,JB)+ 0.5)
               ARF_K(JL,JB) = ARF_K(JL,JB)*ARF_N(JL,JB)/NDUM
               ARF_N(JL,JB) = NDUM
            ENDIF
c
         ELSE IF (IPARA.EQ.11) THEN
c
c           *** Overlandflow
            XAREA = AREA(JB)
            XSLOPE = SLOPE(JL, JB)
            XDD = FDD(JL, JB)
            IOMOD = 6
            XN = ALF_N0
            CALL MODTIME(IOMOD, TAU, XN, DX, XAREA, XSLOPE, XDD,IQUE)
c
c           *** obere und untere Grenze fuer n
            IF (XN.LT.0.5) XN=0.5
            IF (XN.GE.5.5) XN=5.5
            ALF_N(JL, JB) = XN
            ALF_K(JL, JB) = TAU / ALF_N(JL, JB)
c
c           *** Riverflow
            IRMOD=15
            XN = ARF_N0
            CALL MODTIME(IRMOD, TAU, XN, DX, XAREA, XSLOPE, XDD,IQUE)
c
c           *** obere und untere Grenze fuer n
            IF (XN.LT.0.5) XN=0.5
            IF (XN.GE.10.5) XN=10.5
            ARF_N(JL, JB) = XN
            ARF_K(JL, JB) = TAU / ARF_N(JL, JB)
         ENDIF
c
c        *** n integer ??? gerundet, daher + 0.5 bei Integerberechnung
         IF (IPARA.EQ.2) THEN
            NDUM = INT(ALF_N(JL,JB)+ 0.5)
            ALF_K(JL,JB) = ALF_K(JL,JB)*ALF_N(JL,JB)/NDUM
            ALF_N(JL,JB) = NDUM
c
            NDUM = INT(ARF_N(JL,JB)+ 0.5)
            ARF_K(JL,JB) = ARF_K(JL,JB)*ARF_N(JL,JB)/NDUM
            ARF_N(JL,JB) = NDUM
c
c        *** Lineare Abhaengigkeit von der Topographie-Streuung
         ELSE IF (IPARA.EQ.3) THEN
            ADUM = ALF_N(JL,JB) * TSIG0 / TSIG(JL, JB)
            NDUM = INT(ADUM + 0.5)
c
c           *** obere und untere Grenze fuer n
            IF (NDUM.LT.1) NDUM = 1
            IF (NDUM.GT.5) NDUM = 5
            ALF_K(JL,JB) = ALF_K(JL,JB)*ADUM/FLOAT(NDUM)
            ALF_N(JL,JB) = FLOAT(NDUM)
c
c        *** Tau ~ TAU_old * (QA*TSIG^QB + QC), n ~ sigma
         ELSE IF (IPARA.EQ.4) THEN
            TAU = ALF_N(JL,JB) * ALF_K(JL,JB)
            FDUM = QA * TSIG(JL, JB)**QB + QC
            ADUM = TAU * FDUM
c
c           *** NDUM = INT( (TSIG-SIGMIN)/TSIG0) )+2
            NDUM = INT( (TSIG(JL, JB)-20.)/60. ) + 2
c
c           *** obere und untere Grenze fuer n
            IF (NDUM.LT.1) NDUM = 1
            IF (NDUM.GT.5) NDUM = 5
c
            ALF_K(JL,JB) = ADUM/FLOAT(NDUM)
            ALF_N(JL,JB) = FLOAT(NDUM)
c
c        *** Tau ~ TAU_old * (QA*TSIG^QB + QC), n bleibt gleich
         ELSE IF (IPARA.EQ.5) THEN
            FDUM = QA * TSIG(JL, JB)**QB + QC
            ARF_K(JL,JB) = FDUM * ARF_K(JL,JB)
c
c        *** Quadratische Abhaengigkeit von der Topographie-Streuung
         ELSE IF (IPARA.EQ.6) THEN
            ADUM = ALF_N(JL,JB) * TSIG0/TSIG(JL, JB) *
     &                            TSIG0/TSIG(JL, JB)
            NDUM = INT(ADUM + 0.5)
c
c           *** obere und untere Grenze fuer n
            IF (NDUM.LT.1) NDUM = 1
            IF (NDUM.GT.5) NDUM = 5
            ALF_K(JL,JB) = ALF_K(JL,JB)*ADUM/FLOAT(NDUM)
            ALF_N(JL,JB) = FLOAT(NDUM)
c
c        *** Heavyside-Abhaengigkeit (O.) von der Topographie-Streuung
         ELSE IF (IPARA.EQ.7) THEN
            IF (TSIG(JL, JB).LE.50.78) THEN
               ALF_K(JL,JB) = ALF_K(JL,JB) * 2.
            ELSE
               ALF_K(JL,JB) = ALF_K(JL,JB) * 0.5
            ENDIF
         ENDIF
c
 150  CONTINUE
c
      IF (NIS.NE.0) WRITE(*,*) NIS, " mal wurde alte Sausen-Analogie ",
     &     "mit Normal Slope statt Inner Slope verwendet."
c
c     *** ------------------------------------------------------------------
c     ******** PART 2: Korrekturen durch Lakes and Swamps ******************
c     ********         sowie Baseflowparameterisierung    ******************
c     *** ------------------------------------------------------------------
c     ***
c     *** Belegen von FDD mit Lake-Percentage [%]
c     *** Belegen von SLINN mit Swamp-Percentage [%]
c     *** Belegen von FORO mit Matthews Wetland Types oder Permafrost
c     ***
c     ***
c
c     ******* Input-Dateien-Auslese
c
c     *** Lake Percentage
      CINI = "TDNLAK"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, FDD, NL, NB, IQUE)
c
c     *** Swamp Percentage
      CINI = "TDNSWA"
      CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LU, DNINP, IFIN, SLINN, NL, NB, IQUE)
c
c     *** Development-Parameter
      CINI = "ILAMOD"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      ILAMOD = INT(FDUM+0.01)
      CINI = "VLA100"
      CALL PARINP(LUF, CINI, VLA100, ZEILE, IQUE)
c
      CINI = "ISWMOD"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      ISWMOD = INT(FDUM+0.01)
      CINI = "VSW100"
      CALL PARINP(LUF, CINI, VSW100, ZEILE, IQUE)
c
      CINI = "PROARE"
      CALL PARINP(LUF, CINI, PROARE, ZEILE, IQUE)
c
      IF (ISWMOD.EQ.5) THEN
c
c        *** Matthews Wetland Types
         CINI = "TDNMWT"
         CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
         CALL GLREAD(LU, DNINP, IFIN, FORO, NL, NB, IQUE)
      ELSE IF (ISWMOD.EQ.6) THEN
c
c        *** Permafrost-Daten
         CINI = "TDNPER"
         CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
         CALL GLREAD(LU, DNINP, IFIN, FORO, NL, NB, IQUE)
      ENDIF
c
c     *** Baseflowparameterisierung
      CINI = "IBASE"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IBASE = INT(FDUM+0.01)
      CINI = 'IGMEM'
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IGMEM = INT(FDUM+0.01)
c
c     *** Addition von Lakes and Swamps?
      IF (ISWMOD.EQ.1) THEN
         DO JB=1, NB
         DO JL=1, NL
            FDD(JL,JB) = FDD(JL,JB) + SLINN(JL,JB)
             IF (FDD(JL,JB).GT.100) FDD(JL,JB) = 100.
         ENDDO
         ENDDO
      ENDIF
c
c     ******* Parameter-Korrektur Schleife
c
      NWK0=0
      NWK1=0
      NWK2=0
      DO 250 JB = 1, NB
      DO 250 JL = 1, NL
         IF (FLAG(JL, JB).LT.0.5 .OR. GFLAG(JL, JB).GT.0.5) GOTO 250
c
c        *** IL, IB = relative Richtungskoordinaten
c        *** Die 0.1-Summanden sind noetig wegen Cray-Rundungsungenauigkeiten
         IB = -( INT( (FDIR(JL,JB)-1.)/3. + 0.1) - 1)
         IL = INT( ( (FDIR(JL,JB)+2.)/3. -
     &        INT((FDIR(JL,JB)+2.)/3. + 0.1) ) * 3. + 0.1) -1
c
c        *** Lokale Senke ?
         IF (IL.EQ.0 .AND. IB.EQ.0) GOTO 250
C
         JLNEW = JL + IL
         JBNEW = JB + IB
c
c        *** Greenwichgrenze
         IF (JLNEW.EQ.0) JLNEW = NL
         IF (JLNEW.EQ.NL+1) JLNEW = 1
c
         DX = SQRT(IB*IB*DLAT*DLAT + IL*IL*DLON(JB)*DLON(JB))
ccc         DH = FORO(JL, JB) - FORO(JLNEW, JBNEW)
c
c        *** Anwendung der Multiplikationsfaktoren
         ALF_K(JL,JB) = ALF_K(JL,JB) * FK_LFK
         ALF_N(JL,JB) = ALF_N(JL,JB) * FK_LFN
         IF (ALF_N(JL,JB).LE.0.5) THEN
ccc            NDUM = INT(ALF_N(JL,JB)+ 0.5)
            NDUM = 1
            ALF_K(JL,JB) = ALF_K(JL,JB)*ALF_N(JL,JB)/NDUM
            ALF_N(JL,JB) = NDUM
         ENDIF
c
         ARF_K(JL,JB) = ARF_K(JL,JB) * FK_RFK
         ARF_N(JL,JB) = ARF_N(JL,JB) * FK_RFN
         IF (ARF_N(JL,JB).LE.0.5) THEN
ccc            NDUM = INT(ARF_N(JL,JB)+ 0.5)
            NDUM = 1
            ARF_K(JL,JB) = ARF_K(JL,JB)*ARF_N(JL,JB)/NDUM
            ARF_N(JL,JB) = NDUM
         ENDIF
c
c        ******* Lake Percentage
         IF (FDD(JL,JB).GT.0) THEN
c
c          *** Riverflow
           VDUM = DX / ( ARF_K(JL, JB)*ARF_N(JL, JB)*86400. )
c
c          * nach Charbonneau
           IF (ILAMOD.EQ.1) THEN
             IF (FDD(JL,JB).GE.PROARE) THEN
             IF (VLA100.GE.VDUM)
     &         WRITE(*,*) "JL=", JL, "  JB=", JB,
     &         "  VDUM =", VDUM, "  Riv_k=", ARF_K(JL, JB)

               ADUM = VDUM* (1 - (1 - VLA100/VDUM)**(100./FDD(JL, JB)))
               ARF_N(JL, JB) = 1.
               ARF_K(JL, JB) = DX / ( ADUM*86400. )
             ENDIF
c
c          * tanh-Ansatz
           ELSE IF (ILAMOD.EQ.2) THEN
             AD=(1-VLA100/VDUM)/2.
ccc            ADUM = 1-  AD *( tanh(2*PI *(FDD(JL, JB)-PROARE)*0.01) +1)
             ADUM = 1-  AD *( tanh(4*PI *(FDD(JL, JB)-PROARE)*0.01) +1)
             ARF_N(JL, JB) = 1.
             ARF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )

ccc             IF (FDD(JL,JB).EQ.100)
ccc     &         WRITE(*,*) "JL=", JL, "  JB=", JB,
ccc     &         "  VDUM =", VDUM, "  ADUM=", ADUM,
ccc     &         "  k=", ARF_K(JL, JB)
           ENDIF
c
c          ******* Overland flow
           VDUM = DX / ( ALF_K(JL, JB)*ALF_N(JL, JB)*86400. )
c
c          * nach Charbonneau
           IF (ILAMOD.EQ.1) THEN
             IF (FDD(JL,JB).GE.PROARE) THEN
             CDUM = VLA100/VDUM
               IF (VLA100.GE.VDUM) THEN
                 WRITE(*,*) "JL=", JL, "  JB=", JB,
     &           "  VDUM =", VDUM, "  Over_k=", ALF_K(JL, JB)
                 CDUM = 0.1
               ENDIF
               ADUM = VDUM* (1 - (1 - CDUM)**(100./FDD(JL, JB)))
               ALF_N(JL, JB) = 1.
               ALF_K(JL, JB) = DX / ( ADUM*86400. )
             ENDIF
c
c          * tanh-Ansatz
           ELSE IF (ILAMOD.EQ.2) THEN
c###             AD=(1-VLA100/VDUM)/2.
             AD=(1-VLA100*0.1/VDUM)/2.
c
             IF (VLA100*0.1.GE.VDUM) THEN
                 WRITE(*,*) "Lake: JL=", JL, "  JB=", JB,
     &           "  VDUM =", VDUM, "  Over_k=", ALF_K(JL, JB)
                 AD=(1-0.1)/2.
             ENDIF
             ADUM = 1-  AD *( tanh(4*PI *(FDD(JL, JB)-PROARE)*0.01) +1)
             ALF_N(JL, JB) = 1.
             ALF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
           ENDIF
         ENDIF
c
c        ******* Wetland Percentage *********************************
         IF (ISWMOD.GT.1 .AND. SLINN(JL,JB).GT.0) THEN
c
c          *** Riverflow
           VDUM = DX / ( ARF_K(JL, JB)*ARF_N(JL, JB)*86400. )
           IF (ISWMOD.EQ.2 .AND. VSW100.LT.VDUM) THEN
             AD=(1-VSW100/VDUM)/2.
             ADUM = 1-AD *( tanh(4*PI *(SLINN(JL,JB)-PROARE)*0.01) +1)
             ARF_N(JL, JB) = 1.
             ARF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
           ELSE IF (ISWMOD.EQ.3 .AND. VSW100.LT.VDUM) THEN
             ADUM = 1.- (1.-VSW100/VDUM) * SLINN(JL,JB) * 0.01
             ARF_N(JL, JB) = 1.
             ARF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
c
c          *** Abhaengigkeit vom Wetland Type
           ELSE IF (ISWMOD.EQ.5 .AND. VSW100.LT.VDUM) THEN
             IF (FORO(JL,JB).GE.6) THEN
                AD=0.
ccc               IF (VSW100*2. .GE. VDUM) THEN
ccc                 AD=0.
ccc               ELSE
ccc                 AD=(1-VSW100*2./VDUM)/2.

ccc               ENDIF
             ELSE
               AD=(1-VSW100/VDUM)/2.
             ENDIF
             IF (AD.GT.0.) THEN
               ADUM = 1-AD *(tanh(4*PI *(SLINN(JL,JB)-PROARE)*0.01) +1)
               ARF_N(JL, JB) = 1.
               ARF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
             ENDIF
c
c          *** Abhaengigkeit vom Wetland Type
           ELSE IF (ISWMOD.EQ.6 .AND. FORO(JL,JB).NE.3. .AND.
     &              FORO(JL,JB).NE.2. .AND. VSW100.LT.VDUM) THEN
             AD=(1-VSW100/VDUM)/2.
             ADUM = 1-AD *( tanh(4*PI *(SLINN(JL,JB)-PROARE)*0.01) +1)
             ARF_N(JL, JB) = 1.
             ARF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
           ENDIF
c
c          *** Overland flow
           VDUM = DX / ( ALF_K(JL, JB)*ALF_N(JL, JB)*86400. )
           IF ( (ISWMOD.EQ.2 .OR.ISWMOD.EQ.5 .OR.ISWMOD.EQ.6)) THEN
c
ccc 10.6.97raus     &           .AND. VSW100/10..LT.VDUM) THEN
c#####
ccc             AD=(1-VSW100/10./VDUM)/2.
             IF (VSW100*0.1.LT.VDUM) THEN
                AD=(1-VSW100*0.1/VDUM)/2.
                NWK1=NWK1 + 1
             ELSE
                NWK2=NWK2 + 1
             ENDIF
             ADUM = 1-AD *( tanh(4*PI *(SLINN(JL,JB)-PROARE)*0.01) +1)
             ALF_N(JL, JB) = 1.
             ALF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
           ELSE IF (ISWMOD.EQ.3 .AND. VSW100.LT.VDUM) THEN
             ADUM = 1.- (1.-VSW100/VDUM) * SLINN(JL,JB)* 0.01
             ALF_N(JL, JB) = 1.
             ALF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
           ELSE IF (ISWMOD.EQ.4.AND. VSW100.LT.VDUM) THEN
             AD=(1-VSW100/VDUM)/2.
             ADUM = 1-AD *( tanh(4*PI *(SLINN(JL,JB)-PROARE)*0.01) +1)
             ALF_N(JL, JB) = 1.
             ALF_K(JL, JB) = DX / ( ADUM*VDUM*86400. )
           ELSE
              WRITE(*,*) "ERROR?: Wet: VDUM=", VDUM,
     &                   "  JL=", JL, "  JB=", JB
           ENDIF
         ENDIF
c
c        *** Velocities [Lag in Days ==> Faktor 86400
ccc         IF (DH.EQ.0) GOTO 250
         FLVEL(JL, JB) = DX / ( ALF_K(JL, JB)*ALF_N(JL, JB)*86400. )
         FRVEL(JL, JB) = DX / ( ARF_K(JL, JB)*ARF_N(JL, JB)*86400. )
c
c        ******* Baseflowparameterisierung
         IF (IBASE.EQ.0) THEN
c
c           *** Baseflow (vorlaeufig konstant - so wie bis Bebruar 1997)
            AGF_K(JL,JB) = 300.
         ELSE IF (IBASE.EQ.1) THEN
c
c           *** Baseflow gitterbox-groessenabhaengig [m] / [m] * [day]
            AGF_K(JL,JB) = DX / 50000. * 300.
         ELSE IF (IBASE.EQ.2 .OR. IBASE.EQ.3) THEN
c
c           *** nach Beate Mueller
            BB = (TSIG(JL,JB) - 100.) / (TSIG(JL,JB) + 1000.)
            IF (BB.LT.0.01) BB=0.01
            XIB = 1. - BB + 0.01
            AGF_K(JL,JB) = 300. / XIB
            IF (IBASE.EQ.3) AGF_K(JL,JB) = DX/50000. * AGF_K(JL,JB)
         ELSE IF (IBASE.EQ.4) THEN
c
c           *** new ideas
            BB = (TSIG(JL,JB) - 100.) / (TSIG(JL,JB) + 1000.)
            IF (BB.LT.0.01) BB=0.01
            XIB = 1./(1. + 20*(SQRT(BB)-0.1))
            AGF_K(JL,JB) = DX/50000. * 300. * XIB
         ENDIF
c
         AGF_K(JL,JB) = AGF_K(JL,JB) * FK_GFK
c
c        *** Baseflowspeicherinitialisierung (vorlaeufig konstant ########)
c###         IF (IGMEM.EQ.1) FGSP(JL,JB) = 0.1 * AREA(JB) / 86400.
c
 250  CONTINUE
c
      WRITE(*,*) "Overlandflow: Wetland-Korrektur: VSW100=0.1 VSW100:"
     &         , "NWK1 =", NWK1
      WRITE(*,*) "Overlandflow: No Wet-Korrektur f. VSW100/10 > VDUM:"
     &         , "NWK2 =", NWK2
c
c     *** Schreiben der Globalen Parameterfelder
c
c     ZEILE = "mv over_k.dat over_k.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv over_n.dat over_n.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv riv_k.dat riv_k.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv riv_n.dat riv_n.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv bas_k.dat bas_k.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv over_vel.dat over_vel.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv riv_vel.dat riv_vel.dat~"
c     CALL SYSTEM(ZEILE)
c     ZEILE = "mv slope.dat slope.dat~"
c     CALL SYSTEM(ZEILE)
c
      DNOUT = "over_k.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, ALF_K, NL, NB, IQUE)
      DNOUT = "over_n.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, ALF_N, NL, NB, IQUE)
      DNOUT = "riv_k.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, ARF_K, NL, NB, IQUE)
      DNOUT = "riv_n.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, ARF_N, NL, NB, IQUE)
      DNOUT = "bas_k.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, AGF_K, NL, NB, IQUE)
      DNOUT = "over_vel.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, FLVEL, NL, NB, IQUE)
      DNOUT = "riv_vel.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, FRVEL, NL, NB, IQUE)
      DNOUT = "slope.dat"
      CALL GLWRITE(LU, DNOUT, IFOUT, SLOPE, NL, NB, IQUE)
c###      IF (IGMEM.EQ.1) THEN
c###         ZEILE = "mv bas_reservoir.dat bas_reservoir.dat~"
c###         CALL SYSTEM(ZEILE)
c###         DNOUT = "bas_reservoir.dat"
c###         CALL GLWRITE(LU, DNOUT, IFOUT, FGSP, NL, NB, IQUE)
c###      ENDIF
c
c
c     *** The End
 999  END
c
c****************************************************************************
      SUBROUTINE PARINP(LU, CINI, FINI, ZEILE, IQUE)
c****************************************************************************
c
c     ******** Routine, welche das Auslesen von Initialisierungs-Parameter
c              zur Auswahl der Climagroessenberechnung aus der
c              Initialiesungsdatei PARAGEN.inp vornimmt.
c              Programmierung analog Routine METINP in METEOR.for.
c
c     ******** Version 1.0 - September 1995
c              Programmierung und Entwicklung: Stefan Hagemann
c
c     ***     LU = Logical Unit fuer Dateioeffnung
c     ***   CINI = Suchstring - Kennzeichnet den Parameternamen = 6 ZEICHEN
c     ***          If CINI(1:1) = "T" ==> Text gesucht.
c     ***   FINI = Parameterwert
c     ***  ZEILE = If Textvariable gesucht ==> Textinhalt
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c
c     ********* Typischer Dateiaufbau der Initialisierungsdatei PARAGEN.inp
c     ***
c     ***     CINI1: Kommentar zu FINI1
c     ***     FINI1
c     ***     CINI2: Kommentar zu FINI2
c     ***     FINI2
c     ***       :            :
c     ***       :            :
c
      REAL FINI
      CHARACTER CINI*6, ZEILE*1024
      INTEGER IQUE, IOS
c
c     *** Oeffnen der Initialisierungsdatei
      OPEN(LU,
     &     FILE='paragen.inp',
     &     ACCESS='sequential',FORM='formatted',
     &     STATUS='UNKNOWN', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) '*** Fehler bei Dateioeffnung meteor.inp in PARINP'
         GOTO 999
      ENDIF
c
 100  READ(LU, '(A1024)', END = 900) ZEILE
      IF (INDEX(ZEILE, CINI).NE.0) THEN
         IF (CINI(1:1).EQ."T") THEN
            READ(LU, '(A1024)') ZEILE
            WRITE(*,*) "Initialisierung: ", CINI," = "
            WRITE(*,*) ZEILE
         ELSE
            READ(LU, *) FINI
            WRITE(*,*) "Initialisierung: ", CINI," = ", FINI
         ENDIF
         GOTO 999
      ENDIF
      GOTO 100
c
 900  WRITE(*,*) " "
      WRITE(*,*) "*** ", CINI," wurde nicht gefunden ***"
c
C     *** Schliessen der Datei LU
C
 999  CLOSE(LU, STATUS='KEEP',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateischliessung in PARINP"
      ENDIF
c
c        *** The End
      RETURN
      END
c
