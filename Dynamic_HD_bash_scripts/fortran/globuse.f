c****************************************************************************
c     *******  GLOBUSE.for     Routinen zur globalen Dateiverarbeitung
c     ***
c     *** GLREAD        Einlesen von globalen Datenfelder
c     *** AREAREAD      Einlesen des globalen Flaechen/Abstandsarray
c     *** GLWRITE       Schreiben von globalen Datenfelder
c     *** SRVREAD       Lesen von SRV files
c     *** SRVWRITE      Schreiben von SRV files
c     *** ISOWRITE      Schreiben einer Isodatei
c     *** GLOBINP       Auslese des Inputfiles global.inp
c     ***
c     ***
c****************************************************************************
      SUBROUTINE GLREAD(LU, DNAM, IFORM, FWERT, NL, NB, IQUE)
c****************************************************************************
c
c     ******** Routine zur Auslese von globalen Datenarrays
c              Die Koordinate mit Index (1, 1) bezeichnet die Gitterbox
c              Nordpol-Datumsgrenze, d.h. 89.75 N und -179.75 W
c
c     ******** Version 1.0 - September 1995
c              Programmierung und Entwicklung: Stefan Hagemann
c
c     ******** Version 1.1 - Oktober 1995
c              Beseitigen der Koordinaten-Transformation fuer IFORM = 1
c              Einfuehren von IFORM = 4 hierfuer, auch wenn vermutlich unnoetig.
c
c     ******** Version 1.2 - June 2001
c              Abfangen LU = 0
c
c     ******** Variablenliste:
c     ***
c     ***   DNAM = Name des Datenfiles
c     ***  FWERT = Datenarray FWERT(NL, NB)
c     ***  IFORM = Art des Speicherformats
c     ***          1 = Cray-Binaerformat fuer Cray-Lauf ohne Koord.-Trafo
c     ***          2 = REGEN: Globales Binaerformat
c     ***          3 = REGEN: Waveiso2-Format
c     ***          4 = Cray-Binaerformat mit Koord.-Trafo auf Datumsgrenze
c     ***     NL = Arraygrenze: Anzahl der Laengengrade
c     ***     NB = Arraygrenze: Anzahl der Breitengrade
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c     ***     LU = Logical Unit der zu lesenden Datei
c
      INTEGER IFORM, IQUE, LU, IH(8), IOS
      REAL*4 FWERT(NL, NB), FLON(720)
      CHARACTER DNAM*1024, CFORM*40
c
      CFORM = '(8F8.2)'
c
      IF (LU.LE.0) THEN
         WRITE(*,*) "**** ERROR:  Logical Unit <= 0: ", LU
         STOP
      ENDIF
      IF (IFORM.LE.0) THEN
         WRITE(*,*) "**** ERROR:  IFORM <= 0: ", IFORM
         STOP
      ENDIF
c
c     *** Cray-Binaerformat
      IF (IFORM.EQ.1 .OR. IFORM.EQ.4) THEN
         OPEN(LU, FILE=DNAM, form='unformatted',
     &      status='old', IOSTAT=IOS)
         IF (IOS.NE.0) THEN
            WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
            WRITE(*,*) "******** Errornummer ",IOS
            STOP
         ENDIF
c
         READ(LU) IH
         READ(LU) FWERT

c
c        *** Koordinatentransformation
c        *** L: 1->361, 360->720, 361->1, 720->360
         IF (IFORM.EQ.4) THEN
            DO JB = 1, NB
               DO JL = 1, NL
                  FLON(JL) = FWERT(JL, JB)
               ENDDO
               DO JL = 1, NL
                  JLNEU = JL + INT(NL/2 + 0.001)
                  IF (JLNEU.GT.NL) JLNEU = JLNEU - NL
                  FWERT(JLNEU, JB) = FLON(JL)
               ENDDO
            ENDDO
         ENDIF
c
c        *** REGEN: Globales Binaerformat
      ELSE IF (IFORM.EQ.2) THEN
         OPEN(LU, FILE=DNAM, form='unformatted',
     &      status='old', IOSTAT=IOS, CONVERT='big_endian')
         IF (IOS.NE.0) THEN
            WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
            WRITE(*,*) "******** Errornummer ",IOS
            STOP
         ENDIF
         READ(LU) FWERT
c
c        *** Waveiso-Format
      ELSE IF (IFORM.EQ.3) THEN
         OPEN(LU, FILE=DNAM, form='formatted',
     &      status='old', IOSTAT=IOS)
         IF (IOS.NE.0) THEN
            WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
            WRITE(*,*) "******** Errornummer ",IOS
            STOP
         ENDIF
         READ(LU, CFORM) ((FWERT(JL,JB),JL=1,NL), JB=1,NB)
      ENDIF
      WRITE(*,*) "Es wurde gelesen die Datei:", DNAM
c
 999  CLOSE(LU)
      RETURN
      END
c
c****************************************************************************
      SUBROUTINE AREAREAD(LU, DNAM, AREA, DELTAL, DELB, JBMAX, IQUE)
c****************************************************************************
c
c     ******** Programm zum Auslesen von Flaechen und Abstaenden
c              in einem globalem Gitternetz. Der globale Laengenfile
c              befindet sich auf der Cray. Dieser muss jedoch noch so
c              umgeformt werden, dass er fuer die Gitterboxlaengen selbst
c              zutrifft, und nicht fuer deren Mittelpunktsabstaende.
c              JBMAX = Anzahl der Breitengrade
c              Aufloesung: 0.5 * 0.5 Grad ==> JBMAX=360
c
c              Da DELTAB(JB)=const., macht dieses keine Schwierigkeiten.
c              Da DELTAL(JB) unabhaengig von JL ist, ist Umformung unnoetig
c
c     *** Globaler Laengen-File Cray: /mf/m/m214003/RUNOFF/30min/fl_dp_dl
c     *** Globaler Laengen-File Regen: /mf/m/mo/m214046/data/fl_dp_dl.dat
c
c     ******** Version 1.0 - November 1994
c              Programmierung und Entwicklung: Stefan Hagemann
c
c     ******** Variablenliste:
c     ***
c     ***  JBMAX = Groesse der Array-y-Achse. i.allg. Breitenkoordinate
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c     ***     LU = Logical Unit der zu lesenden Datei
c     ***   AREA(JB) = Feld der Gitterboxflaechen, Einheit = [m^2]
c     *** DELTAB(JB) = Feld der latitudinalen Abstaende zwischen den
c     ***              Gitterboxmittelpunkten = Constant = DELB
c     ***   DELB = Konstanter latitudinaler Abstand, Einheit = [m]
c     *** DELTAL(JB) = Feld der longitudinalen Abstaende zwischen den
c     ***              Gitterboxmittelpunkten, Einheit = [m]
c
c     *** Feldgroesse
ccc      PARAMETER(JBMAX=360)
c
ccc      REAL AREA(JBMAX), DELTAL(*), DELTAB(JBMAX-1)
      REAL AREA(*), DELTAL(*)
      CHARACTER DNAM*1024
c
c     *** Auslesen des globalen Laengen-Files DNAM
      OPEN(LU, FILE=DNAM,CONVERT='big_endian')

      READ(LU,*) (AREA(I),I=1,JBMAX)
      READ(LU,*) (DELTAL(I),I=1,JBMAX-1)
      DELB=DELTAL(2)
      READ(LU,*) (DELTAL(I),I=1,JBMAX)
      CLOSE(LU)
      WRITE(*,*) "*** Laengenfelder ausgelesen aus ",DNAM
c
c
      IF (IQUE.NE.0) THEN
         WRITE(*,*) (AREA(I),I=1,JBMAX)
         WRITE(*,*) DELB
         WRITE(*,*) (DELTAL(I),I=1,JBMAX)
      ENDIF
c
      RETURN
      END
c
c****************************************************************************
      SUBROUTINE GLWRITE(LU, DNAM, IFORM, FWERT, NL, NB, IQUE)
c****************************************************************************
c
c     ******** Routine zum Schrieben von globalen Datenarrays
c              Die Koordinate mit Index (1, 1) bezeichnet die Gitterbox
c              Nordpol-Datumsgrenze, d.h. 89.75 N und -179.75 W
c
c     ******** Version 1.0 - September 1995
c              Programmierung und Entwicklung: Stefan Hagemann
c
c     ******** Version 1.1 - Oktober 1995
c              Beseitigen der Koordinaten-Transformation fuer IFORM = 1
c              Einfuehren von IFORM = 4 hierfuer, auch wenn vermutlich unnoetig.
c
c     ******** Variablenliste:
c     ***
c     ***   DNAM = Name des Datenfiles
c     ***  IFORM = Art des Speicherformats
c     ***          1 = Cray-Binaerformat fuer Cray-Lauf ohne Koord.-Trafo
c     ***          2 = REGEN: Globales Binaerformat
c     ***          3 = REGEN: Waveiso2-Format
c     ***          4 = Cray-Binaerformat mit Koord.-Trafo von Datumsgrenze
c     ***  FWERT = Datenarray FWERT(NL, NB)
c     ***     NL = Arraygrenze: Anzahl der Laengengrade
c     ***     NB = Arraygrenze: Anzahl der Breitengrade
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c     ***     LU = Logical Unit der zu lesenden Datei
c
      INTEGER IFORM, IQUE, LU, IH(8), IOS
      REAL FWERT(NL, NB), FLON(720)
      CHARACTER DNAM*1024, CFORM*40
c
      CFORM = '(8F8.2)'
c
c     *** Cray-Binaerformat
      IF (IFORM.EQ.1 .OR. IFORM.EQ.4) THEN
         DO IOS=1,8
           IH(IOS)=0
         ENDDO
         OPEN(LU, FILE=DNAM, form='unformatted', IOSTAT=IOS)
         IF (IOS.NE.0) THEN
            WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
            WRITE(*,*) "******** Errornummer ",IOS
            GOTO 999
         ENDIF
c
c        *** Koordinaten-Ruecktransformation
c        *** L: 361->1, 720->360, 1->361, 360->720
         IF (IFORM.EQ.4) THEN
            DO JB = 1, NB
               DO JL = 1, NL
                  FLON(JL) = FWERT(JL, JB)
               ENDDO
               DO JL = 1, NL
                  JLNEU = JL + INT(NL/2 + 0.001)
                  IF (JLNEU.GT.NL) JLNEU = JLNEU - NL
                  FWERT(JLNEU, JB) = FLON(JL)
               ENDDO
            ENDDO
         ENDIF
c
         IH(5) = NL
         IH(6) = NB
         WRITE(LU) IH
         WRITE(LU) FWERT
c
c        *** REGEN: Globales Binaerformat
      ELSE IF (IFORM.EQ.2) THEN
         OPEN(LU, FILE=DNAM, form='unformatted',
     &      IOSTAT=IOS,CONVERT='big_endian')
         IF (IOS.NE.0) THEN
            WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
            WRITE(*,*) "******** Errornummer ",IOS
            GOTO 999
         ENDIF
         WRITE(LU) FWERT
c
c        *** Waveiso-Format
      ELSE IF (IFORM.EQ.3) THEN
         OPEN(LU, FILE=DNAM, form='formatted',
     &      status='new', IOSTAT=IOS)
         IF (IOS.NE.0) THEN
            WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
            WRITE(*,*) "******** Errornummer ",IOS
            GOTO 999
         ENDIF
         WRITE(LU, CFORM) ((FWERT(JL,JB),JL=1,NL), JB=1,NB)
      ENDIF
c
      WRITE(*,*) "Es wurde geschrieben die Datei:", DNAM
c
 999  CLOSE(LU)
      RETURN
      END
c
c***********************************************************************
      SUBROUTINE ISOWRITE(LU, DNAM, F1, F2, NMES)
c***********************************************************************
C
C     ******** Unterprogramm zum Schreiben von reinen Iso-Dateien
c              ohne Kommentarzeilen, den Werten F1(*) in der ersten
c              Spalte und den Werten F2(*) in der zweiten Spalte.
c
c     ******** Version 1.0  -  1.95  -  by Hag
c
      INTEGER NMES, LU
      REAL F2(*), F1(*)
      CHARACTER DNAM*1024
C
C     ********* Oeffnen der Datei DNAM
c
      OPEN(LU,FILE=DNAM, FORM='formatted', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) '********** Fehler bei Dateioeffnung in ISOWRITE'
         GOTO 599
      ENDIF
C
      DO 550 I=1,NMES
         WRITE(LU, *) F1(I), F2(I)
  550 CONTINUE
C
      WRITE(*,'(A,A12,A)') '*** Datei ',DNAM,'wurde geschrieben ***'
c
C     ************ Schliessen der Datei 1 ******************************
C
  599 CLOSE (LU,STATUS='KEEP',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) '******** Fehler bei Dateischliessung in ISOWRITE'
         GOTO 999
      ENDIF
c
  999 RETURN
      END
c
c
c****************************************************************************
      SUBROUTINE GLOBINP(LU, DNAM, CINI, FINI, ZEILE, IQUE)
c****************************************************************************
c
c     ******** Routine, welche das Auslesen von Initialisierungs-Parameter
c              und Dateinamen fuer die Abfluss-Simulation aus der
c              Initialisierungsdatei GLOBAL.inp vornimmt.
c              Programmierung analog Routine PARINP in PARAGEN.for.
c
c     ******** Version 1.0 - September 1995
c              Programmierung und Entwicklung: Stefan Hagemann
c
c     ******** Version 1.1 - Maerz 1996
c              Einbau der Uebergabe des Inputdateinamens DNAM
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
      CHARACTER CINI*6, ZEILE*1024, DNAM*1024
      INTEGER IQUE, IOS
c
c     *** Oeffnen der Initialisierungsdatei
      OPEN(LU,FILE=DNAM,ACCESS='sequential',FORM='formatted',
     &     STATUS='UNKNOWN', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) '*** Fehler Input-Dateioeffnung in GLOBINP'
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
         WRITE(*,*) "***** Fehler bei Dateischliessung in GLOBINP"
      ENDIF
c
c        *** The End
      RETURN
      END
c
c
c****************************************************************************
      SUBROUTINE SRVWRITE(LU, DNAM, ICODE, FWERT, NL, NB, IDATE)
c****************************************************************************
c
c     ******** Routine zum Schrieben von SRV files
c
c     ******** Version 1.0 - March 2009
c              Programmierung und Entwicklung: Stefan Hagemann
c              Analog GLWRITE
c
c     ******** Variablenliste:
c     ***
c     ***   DNAM = Name des Datenfiles
c     ***  ICODE = Codenummer
c     ***  IDATE = Datum
c     ***  FWERT = Datenarray FWERT(NL, NB)
c     ***     NL = Arraygrenze: Anzahl der Laengengrade
c     ***     NB = Arraygrenze: Anzahl der Breitengrade
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c     ***     LU = Logical Unit der zu lesenden Datei
c
      INTEGER ICODE, IDATE, LU, IH(8), IOS
      REAL FWERT(NL, NB)
      CHARACTER DNAM*1024
c
      IH(:)=0
      IH(1) = ICODE
      IH(3) = IDATE
      IH(5) = NL
      IH(6) = NB
      OPEN(LU, FILE=DNAM, form='unformatted', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
         WRITE(*,*) "******** Errornummer ",IOS
         GOTO 999
      ENDIF
      WRITE(LU) IH
      WRITE(LU) FWERT
c
      WRITE(*,*) "Es wurde geschrieben die Datei:", DNAM
c
 999  CLOSE(LU)
      RETURN
      END
c
c
c****************************************************************************
      SUBROUTINE SRVREAD(LU, DNAM, ICODE, FWERT, NL, NB)
c****************************************************************************
c
c     ******** Routine zum Lesen von SRV files
c
c     ******** Version 1.0 - Sep 2014
c              Programmierung und Entwicklung: Stefan Hagemann
c              Analog GLWRITE
c
c     ******** Variablenliste:
c     ***
c     ***   DNAM = Name des Datenfiles
c     ***  ICODE = Codenummer
c     ***  FWERT = Datenarray FWERT(NL, NB)
c     ***     NL = Arraygrenze: Anzahl der Laengengrade
c     ***     NB = Arraygrenze: Anzahl der Breitengrade
c     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
c     ***     LU = Logical Unit der zu lesenden Datei
c
      INTEGER ICODE, IDATE, LU, IH(8), IOS
      REAL FWERT(NL, NB)
      CHARACTER DNAM*1024
      OPEN(LU, FILE=DNAM, form='unformatted', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNAM
         WRITE(*,*) "******** Errornummer ",IOS
         STOP
      ENDIF
      READ(LU) IH
      READ(LU) FWERT
      ICODE = IH(1)
      WRITE(*,*) "Es wurde gelesen die Datei:", DNAM
c
 999  CLOSE(LU)
      RETURN
      END
c

