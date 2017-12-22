C**********************************************************************
c*****             MATHE.for                                        ***
c*****                                                              ***
c***** Sammlung mathematischer Subroutinen mit folgendem Inhalt:    ***
c*****                                                              ***
c*****   LINREG   Lineare Regression: Koeffizienten-Berechnung      ***
c*****   MITTEL   Mittelwert-, Streuungs-, Min- und Max-Berechnung  ***
c*****  MONATMIT  Monatsmittelwerte und -Streuungen sowie Annuals   ***
c*****     SORT   Sortieren eines REAL-Feldes                       ***
c*****   SORT_2   Sortieren zweier REAL-Felder in Abh. von Feld 1   ***
c*****  SORTINT   Sortieren eines INTEGER-Feldes                    ***
c*****  CROSSKOR  Kreuz-Korrelation (Lag-Korrelation)               ***
c*****   DAYPLUS  Tageszaehler, der um einen Tag weiter zaehlt      ***
c*****   PARABEL  Programm, das durch drei Punkte eine Parabel legt ***
c*****  HYPERBEL  Programm, das durch 3 Punkte eine Hyperbel legt   ***
c*****     BOLIC  Programm, das durch 3 Punkte die Fkt. aX^p+c legt ***
c*****    MEDIAN  Berechnen des Medians eines REAL-Feldes           ***
c*****   ROOTMSE  Berechnen des RSME zwischen zwei Zeitreihen       ***
c*****                                                              ***
C**********************************************************************
      SUBROUTINE LINREG(X,Z,N,A,B,FZ,FA,FB)
C**********************************************************************
C
c     Weiterentwicklung aus LINREGSR ( copyright by Hag ,17.10.91 )
c     Entspricht der Subroutine LINREGS2 (Version 2.0 - 26.8.92 - by Hag)
c
C     Dieses Unterprogramm fuehrt eine lineare Regression durch.
c     Es uebernimmt die Koordinaten der x-Achse als REAL*4-FELD X und
c     die Koordinaten der z-Achse als REAL*4-FELD Z ,sowie die Anzahl
c     der Koordinatenpaare als INTEGER*4-ZAHL N.
c     Es liefert als Steigung der Regressionsgeraden die REAL*4-ZAHL B
c     und als Schnittpunkt mit der Z-Achse die REAL*4-ZAHL A.
C     Ausserdem liefert es die Standardabweichung fuer die einzelnen
c     Messungen Z(I), sowie fuer B und A in Form der REAL*4-Zahlen FZ,
c     FB und FA.

C     ******************* Vereinbarungsteil ****************************
C
      INTEGER N, I
      REAL X(*),Z(*)
      REAL A,B,SX,SZ,SXZ,SX2,SV2,FZ,FA,FB
C
      SX=0.
      SZ=0.
      SXZ=0.
      SX2=0.
      SV2=0.
C
C     ***************** Summierungsschleife ****************************
C
      DO 10 I=1,N
         SX= SX+X(I)
         SZ= SZ+Z(I)
         SXZ= SXZ+(X(I)*Z(I))
         SX2= SX2+(X(I)*X(I))
   10 CONTINUE
C
C     ******** Berechnung von Steigung B und Z-Achsenschniitpunkt A ****
C
      A= (SX2*SZ - SX*SXZ)/(N*SX2 - SX**2)
      B= (N*SXZ - SX*SZ)/(N*SX2 - SX**2)
C
C     ******** Berechnung der Standardabweichungen *********************
c
      DO 20 I=1,N
         SV2= SV2+(A+B*X(I)-Z(I))**2
   20 CONTINUE
C
      FZ= (SV2/(N-2))**0.5
      FA= FZ*(SX2/(N*SX2-SX**2))**0.5
      FB= FZ*(N/(N*SX2-SX**2))**0.5

      RETURN
      END
c
C**********************************************************************
      SUBROUTINE MITTEL(F,NMES,FXMIT,FXSIG,XMIN,XMAX,LU,GAUS,IQUE)
C**********************************************************************
c
c     ******** Subroutine zur Mittelwertberechnung analog MITTELSR.for
c              Wenn LU <> 0 geschieht Ausgabe von Mittelwert,Streuung,
c              Minimum und Maximum in Datei mit Logical Unit LU
c
c     ******** Version 1.1 - 8.3.93 - by Hag
c
c     ******** Version 1.2 - 15.3.93 - by Hag
c              Einfuegen der Standardabweichung FXSIG in CHIPLO-Block
c
c     ******** Version 2.0 - 30.6.94 - by Hag
c              ohne Common-Block und Umbenennung in MITTEL.for
c
c     ******** Version 2.1 - 13.01.95 - by Hag
c              IF F(I) = -9999. ==> Wert wird nicht beruecksichtigt
c              bei nicht gaussverteilten Werten, d.h. bei GAUS=.false.
c
c     ******** Version 2.2 - 03.95 - by Hag
c              IF Lu = -1 ==> Gar keine Ausgabe der Berechneten Werte,
c              weder in Logfile noch auf Bildschirm.
C
c     *** F(NMES) = ist die Gesamtheit der uebergebenen Messwerte
c     ***    NMES = ist die Anzahl der uebergebenen Messwerte
c     ***    NREL = Relevante Anzahl der Messwerte ohne -9999.-Werte
C     ***    GAUS = "Wird Gaussverteilung erwartet ?"-Variable
C     ***   FXMIT = Mittelwert der Messgroessen
C     ***   FXSIG = Streuung der Messgroessen
c     ***      LU = Logical Unit, wenn 0 ==> Keine Logfileausgabe
c     ***           -1 ==> auch keine Bildschirm-Ausgabe
c     ***    XMIN = Messwertminimum
c     ***    XMAX = Messwertmaximum
C     ***    IQUE = "Kommentar an/aus"-Variable  ( 0 = Kommentar aus )
c     ***           = 10 ==> Nur zum Testen eine Extra-Ausgabe
c
c     **********     Variablen fuer Zeitmessung   *****************
c
c     **************** Vereinbarungsteil ************************************
C
      REAL*4 XSUM,X2,GSUM,GXMIT,GSXMIT,SG2,XDUM
      REAL*4 F(*)
      INTEGER*4 I,IQUE,NGO,NGG, NREL
      LOGICAL GAUS
C
C     ******* Anfangsbelegung **************************************
c
   10 XMIN=9.E+19
      XMAX=-9.E+19
      XSUM=0.
      X2=0.
      FXMIT=0.
      FXSIG=0.
      NREL=NMES
      NGO=0
      GSUM=0.
      SG2=0.
c
C
C     *** Routine zur Mittelwertberechnung
c
      DO 40 I=1,NMES
c
c        *** -9999.-Werte rausschmeissen
         IF (F(I).EQ.-9999.) THEN
            NREL=NREL-1
            GOTO 40
         ENDIF
c
         XSUM=XSUM+F(I)
         IF(F(I).LT.XMIN) XMIN=F(I)
         IF(F(I).GT.XMAX) XMAX=F(I)
         X2=X2+(F(I)*F(I))
       IF(IQUE.EQ.10) WRITE(*,*) 'F= ',F(I),'  X2= ',X2,'  I=',I
   40 CONTINUE
c
      IF (NREL.EQ.0) THEN
         FXMIT=-9999.
      ELSE
         FXMIT=XSUM/FLOAT(NREL)
      ENDIF
C     *** Testausgabe
      IF(IQUE.EQ.10) WRITE(*,*) '  X2=',X2,'  FXMIT^2=',FXMIT*FXMIT
C
      IF (NREL.EQ.0) THEN
         FXSIG=-9999.
         GOTO 45
      ELSE
         XDUM=X2/FLOAT(NREL)-FXMIT*FXMIT
      ENDIF
C
      IF (XDUM.LE.0) THEN
         IF (XDUM.EQ.0) THEN
            IF (IQUE.NE.0)
     &         WRITE(*,*) 'X^2_MIT - XMIT^2 = 0 !!!!!!!!!!!'
            FXSIG=XDUM
            GOTO 45
         ENDIF
         XDUM=-XDUM
         WRITE(*,*) 'X^2_MIT - XMIT^2 < 0 !!!!!!!!!!!'
      ENDIF
C
      FXSIG=(XDUM)**0.5
C       *** Testausgabe
  45  IF(IQUE.EQ.10) WRITE(*,*) 'Bis hier ist alles o.k. !!!!!!'
C
      IF(GAUS) THEN
         XMIN=FXMIT-3.5*FXSIG
         XMAX=FXMIT+3.5*FXSIG+1
         IF(XMIN.LT.0) XMIN=0.
      ENDIF
C
      IF(GAUS) THEN
         DO 60 I=1,NMES
            IF(F(I).LT.XMIN.OR.F(I).GT.XMAX) THEN
               NGO=NGO+1
               GSUM=GSUM+F(I)
               SG2=SG2+F(I)*F(I)
            ENDIF
   60    CONTINUE
         NGG=NMES-NGO
         GXMIT=(XSUM-GSUM)/NGG
         GSXMIT=((X2-SG2)/NGG-GXMIT*GXMIT)**0.5
      ENDIF
c
      IF (NREL.NE.NMES) WRITE(*,*) NMES-NREL,
     &    "ungueltige Werte wurden rausgeschmissen!"
C
CCC      IF(IQUE.EQ.0) GOTO 100
      IF(GAUS) WRITE(*,*) 'Min. und Max. entsprechen FXMIT +/-3.5*FXSIG'
      IF (LU.EQ.0) THEN
         WRITE(*,*) '  Der Mittelwert ist: ',FXMIT
         WRITE(*,*) 'Die Standardabweichung ist: ',FXSIG
         WRITE(*,*) 'Der Minimumswert ist: ',XMIN
         WRITE(*,*) 'Der Maximumswert ist: ',XMAX
      ELSE IF (LU.EQ.-1) THEN
         GOTO 100
      ELSE
         WRITE(LU,*) '      Der Mittelwert: ',FXMIT
         WRITE(LU,*) 'Die Standardabweichung ist: ',FXSIG
         WRITE(LU,*) 'Der Minimumswert ist: ',XMIN
         WRITE(LU,*) 'Der Maximumswert ist: ',XMAX
      ENDIF
c
c
  100 RETURN
  999 END
c
c
C**********************************************************************
      SUBROUTINE MONATMIT(F, NMES, FMIT, FSIG, IQUE)
C**********************************************************************
c
c
c     ******** Subroutine zur Berechnung von Monats-Mittelwerten und -Streuungen
c              unter der Verwendung von Routine MITTEL
c
c     ******** Version 1.0 - 03.95 - by Hag
c
C
c     *** F(NMES)  = ist die Gesamtheit der uebergebenen Messwerte
c     ***    NMES  = Anzahl der uebergebenen Messwerte des Jahres = 365 oder 366
c     *** NDAY(12) = Anzahl der Tage im Monat
C     *** FMIT(13) = 12 Monats-Mittelwerte der Messgroessen + Annual Mean
C     *** FSIG(13) = 12 Monats-Streuung der Messgroessen + Annual Streuung
C     ***    IQUE  = "Kommentar an/aus"-Variable  ( 0 = Kommentar aus )
c     ***             = 10 ==> Nur zum Testen eine Extra-Ausgabe
c
c     ******* Dummys fuer Routine MITTEL
c     ***
C     ***    GAUS = "Wird Gaussverteilung erwartet ?"-Variable
c     ***    XMIT = Mittelwert
c     ***    XSIG = Streuung
c     ***    XMIN = Messwertminimum
c     ***    XMAX = Messwertmaximum
c     ***      LU = Logical Unit, wenn 0 ==> Keine Logfileausgabe
c     ***           -1 ==> auch keine Bildschirm-Ausgabe
c
c
c
c     **************** Vereinbarungsteil ************************************
C
      REAL*4 XMIT, XSIG, XMIN, XMAX
      REAL*4 F(*), FMON(31), FMIT(13), FSIG(13)
      INTEGER*4 I, J, LU, IQUE, NMES, NDAY(12), NDUM, NSUM
      LOGICAL GAUS
      SAVE NDAY
C
C     ******* Anfangsbelegung **************************************
c
      DATA NDAY /31, 28, 31, 30, 31, 30,  31, 31, 30, 31, 30, 31/
      GAUS = .FALSE.
      LU=-1
      NSUM =0.
c
      IF (NMES.EQ.365) THEN
         NDAY(2)=28
      ELSE IF (NMES.EQ.366) THEN
         NDAY(2)=29
      ELSE
         WRITE(*,*) "***** Fehler: Messwertanzahl <> 365 bzw. 366"
         GOTO 999
      ENDIF
c
c     *** Schleife ueber die Monate
      DO 100 J = 1, 12
         NDUM = NDAY(J)
c
c        *** Schleife ueber die Tage im Monat
         DO I=1, NDUM
            FMON(I) = F(NSUM+I)
         ENDDO
         CALL MITTEL(FMON, NDUM, XMIT,XSIG, XMIN,XMAX, LU, GAUS, IQUE)
         FMIT(J) = XMIT
         FSIG(J) = XSIG
         NSUM = NSUM + NDUM
 100  CONTINUE
C
C     *** Berechnung des Jahresmittelwertes
      CALL MITTEL(F, NMES, XMIT,XSIG, XMIN,XMAX, LU, GAUS, IQUE)
      FMIT(13) = XMIT
      FSIG(13) = XSIG
c
  999 RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE SORT(FELD,IANZ,IAB)
C**********************************************************************
C
C     SORTIERPROGRAMM fuer REAL-Felder analog zu SORTINT.for
C       Sortiert ein Feld von Typ FELD(IANZ)
c
c       IAB = 0   Aufsteigend
c       IAB = 1   Absteigend
c
C       Version 1.0 - 30.6.94 - by Hag
C
C
      DIMENSION FELD(*)
C
Cc
      IF (IAB.EQ.0) GOTO 2000
c
c     *********** Absteigend
c
      DO 1000 IANF = 1,IANZ - 1
C
CC  SUCHEN MAXIMUM VON IANZ AN
        FMAX = -90000.
        DO 500 I = IANF,IANZ
          IF((FELD(I)-0.001).GT.FMAX) THEN
C       * NEUES X GROESSER
                FMAX = FELD(I)
                IMAX = I
          ENDIF
500     CONTINUE
C
CC  MAXIMUM AN POSITION IANZ SETZEN, WENN NICHT SCHON DORT
        IF(IMAX.NE.IANF) THEN
C
CC  FUER 4 FELDER
C         * ZWISCHENSPEICHERN DER VARIABLEN AN DER ANFANGSPOSITION
            DUMMY  = FELD(IANF)
C         * MAXIMUMSVARIABLE  AN DIE ANFANGSPOSITION
            FELD(IANF) = FELD(IMAX)
C         * ZWISCHENSPEICHER AN DIE VORHERIGE MAXIMUMSPOSITION
            FELD(IMAX) =  DUMMY
        ENDIF
C
1000  CONTINUE
      GOTO 9900
c
c     ******** Aufsteigend
c
2000  DO 3000 IANF = 1,IANZ - 1
C
CC  SUCHEN Minimum VON IANZ AN
        FMIN = 900000.
        DO 2500 I = IANF,IANZ
          IF((FELD(I)).LT.FMIN) THEN
C       * NEUES X kleiner
                FMIN = FELD(I)
                IMIN = I
          ENDIF
2500     CONTINUE
C
CC  MINIMUM AN POSITION IANZ SETZEN, WENN NICHT SCHON DORT
        IF(IMIN.NE.IANF) THEN
C
CC  FUER 4 FELDER
C         * ZWISCHENSPEICHERN DER VARIABLEN AN DER ANFANGSPOSITION
            DUMMY  = FELD(IANF)
C         * MINIMUMSVARIABLE  AN DIE ANFANGSPOSITION
            FELD(IANF) = FELD(IMIN)
C         * ZWISCHENSPEICHER AN DIE VORHERIGE MINIMUMSPOSITION
            FELD(IMIN) =  DUMMY
        ENDIF
C
3000  CONTINUE

9900  RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE SORT_2(FELD1, FELD2, IANZ,IAB)
C**********************************************************************
C
C     SORTIERPROGRAMM fuer zwei REAL-Felder, die nach Feld1 sortiert
c       werden. Programmierung analog zu SORT.for.
C       Sortiert zwei Felder von Typ FELD1(IANZ), FELD2(IANZ) nach FELD1
c
c       IAB = 0   Aufsteigend
c       IAB = 1   Absteigend
c
C       Version 1.0 - 12.94 - by Hag
C
C
      DIMENSION FELD1(*), FELD2(*)
C
Cc
      IF (IAB.EQ.0) GOTO 2000
c
c     *********** Absteigend
c
      DO 1000 IANF = 1,IANZ - 1
C
CC  SUCHEN MAXIMUM VON IANZ AN
        FMAX = -90000.
        DO 500 I = IANF,IANZ
          IF((FELD1(I)-0.001).GT.FMAX) THEN
C       * NEUES X GROESSER
                FMAX = FELD1(I)
                IMAX = I
          ENDIF
500     CONTINUE
C
CC  MAXIMUM AN POSITION IANZ SETZEN, WENN NICHT SCHON DORT
        IF(IMAX.NE.IANF) THEN
C
CC  FUER 4 FELDER
C         * ZWISCHENSPEICHERN DER VARIABLEN AN DER ANFANGSPOSITION
            DUMMY  = FELD1(IANF)
            DUM2 = FELD2(IANF)
c
C         * MAXIMUMSVARIABLE  AN DIE ANFANGSPOSITION
            FELD1(IANF) = FELD1(IMAX)
            FELD2(IANF) = FELD2(IMAX)
C         * ZWISCHENSPEICHER AN DIE VORHERIGE MAXIMUMSPOSITION
            FELD1(IMAX) =  DUMMY
            FELD2(IMAX) =  DUM2
        ENDIF
C
1000  CONTINUE
      GOTO 9900
c
c     ******** Aufsteigend
c
2000  DO 3000 IANF = 1,IANZ - 1
C
CC  SUCHEN Minimum VON IANZ AN
        FMIN = 900000.
        DO 2500 I = IANF,IANZ
          IF((FELD1(I)).LT.FMIN) THEN
C       * NEUES X kleiner
                FMIN = FELD1(I)
                IMIN = I
          ENDIF
2500     CONTINUE
C
CC  MINIMUM AN POSITION IANZ SETZEN, WENN NICHT SCHON DORT
        IF(IMIN.NE.IANF) THEN
C
CC  FUER 4 FELDER
C         * ZWISCHENSPEICHERN DER VARIABLEN AN DER ANFANGSPOSITION
            DUMMY  = FELD1(IANF)
            DUM2 = FELD2(IANF)
C         * MINIMUMSVARIABLE  AN DIE ANFANGSPOSITION
            FELD1(IANF) = FELD1(IMIN)
            FELD2(IANF) = FELD2(IMIN)
C         * ZWISCHENSPEICHER AN DIE VORHERIGE MINIMUMSPOSITION
            FELD1(IMIN) =  DUMMY
            FELD2(IMIN) =  DUM2
        ENDIF
C
3000  CONTINUE

9900  RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE SORTINT(IFELD,IANZ,IAB)
C**********************************************************************
C
C     SORTIERPROGRAMM
C       Sortiert ein Feld von Typ IFELD(IANZ)
c
c       IAB = 0   Aufsteigend
c       IAB = 1   Absteigend
c
C       Version 1.0 - 7.93 - by Hag
C
C
      DIMENSION IFELD(*)
C
Cc
      IF (IAB.EQ.0) GOTO 2000
c
c     *********** Absteigend
c
      DO 1000 IANF = 1,IANZ - 1
C
CC  SUCHEN MAXIMUM VON IANZ AN
        JMAX = -9000
        DO 500 I = IANF,IANZ
          IF((IFELD(I)-0.001).GT.JMAX) THEN
C       * NEUES X GROESSER
                JMAX = IFELD(I)
                IMAX = I
          ENDIF
500     CONTINUE
C
CC  MAXIMUM AN POSITION IANZ SETZEN, WENN NICHT SCHON DORT
        IF(IMAX.NE.IANF) THEN
C
CC  FUER 4 FELDER
C         * ZWISCHENSPEICHERN DER VARIABLEN AN DER ANFANGSPOSITION
            DUMMY  = IFELD(IANF)
C         * MAXIMUMSVARIABLE  AN DIE ANFANGSPOSITION
            IFELD(IANF) = IFELD(IMAX)
C         * ZWISCHENSPEICHER AN DIE VORHERIGE MAXIMUMSPOSITION
            IFELD(IMAX) =  DUMMY
        ENDIF
C
1000  CONTINUE
      GOTO 9900
c
c     ******** Aufsteigend
c
2000  DO 3000 IANF = 1,IANZ - 1
C
CC  SUCHEN Minimum VON IANZ AN
        JMIN = 9000
        DO 2500 I = IANF,IANZ
          IF((IFELD(I)).LT.JMIN) THEN
C       * NEUES X kleiner
                JMIN = IFELD(I)
                IMIN = I
          ENDIF
2500     CONTINUE
C
CC  MINIMUM AN POSITION IANZ SETZEN, WENN NICHT SCHON DORT
        IF(IMIN.NE.IANF) THEN
C
CC  FUER 4 FELDER
C         * ZWISCHENSPEICHERN DER VARIABLEN AN DER ANFANGSPOSITION
            DUMMY  = IFELD(IANF)
C         * MINIMUMSVARIABLE  AN DIE ANFANGSPOSITION
            IFELD(IANF) = IFELD(IMIN)
C         * ZWISCHENSPEICHER AN DIE VORHERIGE MINIMUMSPOSITION
            IFELD(IMIN) =  DUMMY
        ENDIF
C
3000  CONTINUE

9900  RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE CROSSKOR(FKX,FKY, NK,NLAG, SS,RK0, RKOR,STAT, IQUE)
C**********************************************************************
c
      REAL FKX(*), FKY(*), RKOR(*)
      INTEGER NK, NLAG, IQUE, LU
      REAL FXMIT, FYMIT, SIGX, SIGY, XMIN, XMAX, RSUM
      LOGICAL GAUS
c
      LU=0
      IF (IQUE.EQ.0) LU=-1
      GAUS=.FALSE.

c
c     *** Mittelwerte
c
      CALL MITTEL(FKX,NK,FXMIT,SIGX,XMIN,XMAX,LU,GAUS,IQUE)
      CALL MITTEL(FKY,NK,FYMIT,SIGY,XMIN,XMAX,LU,GAUS,IQUE)
      SS = SIGY/SIGX
c
c     *** Lag-Korrelation = Cross-Correlation
c
      STAT=0.
      DO 200 L = 0, NLAG
         RSUM = 0.
         DO 100 I=1, NK-L
            RSUM = RSUM + (FKX(I)-FXMIT) * (FKY(I+L)-FYMIT)
  100    CONTINUE
         RSUM = RSUM / (NK * SIGX * SIGY)
         IF (L.EQ.0) THEN
            RK0 = RSUM
         ELSE
            RKOR(L) = RSUM
            STAT = STAT + RSUM*RSUM
         ENDIF
  200 CONTINUE
      STAT = NK * STAT
c
c     *** The End
      RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE DAYPLUS(IJ, IM, ID)
C**********************************************************************
c
c     ******* Routine, welche das Datum, bestehend aus Jahr IJ,
c             Monat IM und Tag ID um einen Tag hochzaehlt.
c             und die betreffenden Werte dieses Tages wieder ausgibt.
c
      REAL NDAY(12)
      SAVE NDAY
C
C     ******* Anfangsbelegung **************************************
c
      DATA NDAY /31, 28, 31, 30, 31, 30,  31, 31, 30, 31, 30, 31/
c
c     *** Schaltjahr ?
      IF ( FLOAT(INT(IJ/4)) .EQ. FLOAT(IJ)/4. ) THEN
         NDAY(2) = 29
      ELSE
         NDAY(2) = 28
      ENDIF
c
c     *** Hochzaehlen
      IF ( ID.EQ.NDAY(IM) ) THEN
         ID = 1
         IF (IM.EQ.12) THEN
            IJ = IJ + 1
            IM = 1
         ELSE
            IM = IM + 1
         ENDIF
      ELSE
         ID = ID + 1
      ENDIF
c
c     *** The End
      RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE PARABEL(X0,Y0, X1,Y1, X2,Y2, A,B,C, IQUE)
C**********************************************************************
c
c     ******* Routine, welche durch die drei Punkte (X0,Y0), (X1,Y1) und
c             (X2,Y2) eine Parabel legt, die definiert ist durch:
c             Y = A*X^2 + B*X + C
c             ==> Ausgabewerte A,B,C
c
c     ******* Version 1.0 -- November 1996 -- by Stefan Hagemann
c
c     *** IQUE = Statusvariable
c     ***      Als Input
c     ***        0 = Normale Berechnung
c     ***        1 = Gleichungssystem mit B = 0 loesen
c     ***            Nur verwenden von Punkt 0 und Punkt 2
c     ***        2 = Gleichungssystem mit A = 0 loesen
c     ***            Nur verwenden von Punkt 0 und Punkt 2
c     ***
c     ***      Als OUTPUT
c     ***        0 = alles in Ordnung
c     ***        1 = Gleichungssystem nicht loesbar
c
      REAL X0, X1, X2, Y0, Y1, Y2, A, B, C
      INTEGER IQUE
c
c
      XQ10 = X1*X1 - X0*X0
      X10  = X1-X0
      XQ20 = X2*X2 - X0*X0
      X20  = X2-X0
      Y10 = Y1-Y0
c
      IF (IQUE.EQ.0) THEN
         A = (Y2-Y0 - Y10*X20/X10 ) / (XQ20 - X20*XQ10/X10)
         B = Y10/X10 - A * XQ10/X10
         C = Y0 - Y10*X0/X10 - A * (X0*X0 - X0 * XQ10/X10)
      ELSE IF (IQUE.EQ.1) THEN
         A = (Y2-Y0 ) / XQ20
         B = 0.
         C = Y0 - X0*X0 * A
      ELSE IF (IQUE.EQ.2) THEN
         A = 0.
         B = (Y2-Y0 ) / X20
         C = Y0 - X0 * B
      ENDIF
c
      IF (A.EQ.B .AND. A.EQ.C) THEN
         IQUE=1
         WRITE(*,*) " GL-System nicht loesbar"
      ELSE
         IQUE=0
      ENDIF
c
c     *** The End
      RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE HYPERBEL(X0,Y0, X1,Y1, X2,Y2, A,B,C, IQUE)
C**********************************************************************
c
c     ******* Routine, welche durch die drei Punkte (X0,Y0), (X1,Y1) und
c             (X2,Y2) eine Hyperbel legt, die definiert ist durch:
c             Y = A*1/X^2 + B*1/X + C
c             ==> Ausgabewerte A,B,C
c
c     ******* Version 1.0 -- November 1996 -- by Stefan Hagemann
c
c
c     *** IQUE = Statusvariable
c     ***      Als Input
c     ***        0 = Normale Berechnung
c     ***        1 = Gleichungssystem mit B = 0 loesen
c     ***            Nur verwenden von Punkt 0 und Punkt 2
c     ***        2 = Gleichungssystem mit A = 0 loesen
c     ***            Nur verwenden von Punkt 0 und Punkt 2
c     ***
c     ***      Als OUTPUT
c     ***        0 = alles in Ordnung
c     ***        1 = Gleichungssystem nicht loesbar
c
      REAL X0, X1, X2, Y0, Y1, Y2, A, B, C
      INTEGER IQUE
c
c
      XQ10 = 1./X1*1./X1 - 1./X0*1./X0
      X10  = 1./X1-1./X0
      XQ20 = 1./X2*1./X2 - 1./X0*1./X0
      X20  = 1./X2-1./X0
      Y10 = Y1-Y0
c
      IF (IQUE.EQ.0) THEN
         A = (Y2-Y0 - Y10*X20/X10 ) / (XQ20 - X20*XQ10/X10)
         B = Y10/X10 - A * XQ10/X10
         C = Y0 - Y10*1./X0/X10 - A * (1./X0*1./X0 - 1./X0 * XQ10/X10)
      ELSE IF (IQUE.EQ.1) THEN
         A = (Y2-Y0 ) / XQ20
         B = 0.
         C = Y0 - 1./(X0*X0) * A
      ELSE IF (IQUE.EQ.2) THEN
         A = 0.
         B = (Y2-Y0 ) / X20
         C = Y0 - 1./(X0) * B
      ENDIF
c
c
      IF (A.EQ.B .AND. A.EQ.C) THEN
         IQUE=1
         WRITE(*,*) " GL-System nicht loesbar"
      ELSE
         IQUE=0
      ENDIF
c
c     *** The End
      RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE BOLIC(X0,Y0, X1,Y1, X2,Y2, A,B,C, IQUE)
C**********************************************************************
c
c     ******* Routine, welche durch die drei Punkte (X0,Y0), (X1,Y1) und
c             (X2,Y2) eine Hyperbolische Funktion legt, die definiert ist durch:
c             Y = A*X^B  + C , p <> 0
c             ==> Ausgabewerte A,B,C
c
c     ******* Version 1.0 -- November 1996 -- by Stefan Hagemann
c
c
c     *** IQUE = Statusvariable
c     ***      Als Input
c     ***        1 = B > 0
c     ***        2 = B < 0
c     ***
c     ***
c     ***      Als OUTPUT
c     ***        0 = alles in Ordnung
c     ***        1 = Gleichungssystem nicht loesbar
c
      REAL X0, X1, X2, Y0, Y1, Y2, A, B, C
      INTEGER IQUE
c
c
c
      IF (IQUE.EQ.1) THEN
         BFAK = 1
      ELSE IF (IQUE.EQ.2) THEN
         BFAK = -1
      ENDIF
c
      YY = (Y1-Y0) / (Y2-Y0)
      DSTEP = 0.1
c
c     *** Suchen des Exponenten B durch simples Nullstellensuchen
      BOLD=0.
 10   Q=0.
      P=0.
      DO I=1, 100
         QOLD = Q
         POLD = P
         P = BFAK * FLOAT(I) * DSTEP + BOLD
         Q = (X1**P - X0**P) / (X2**P - X0**P) - YY
         IF (Q.EQ.0) THEN
            WRITE(*,*) "Nullst.: I=", I, "  P=",P,
     &          "  Q = ", QOLD, " --> Q = ", Q
            GOTO 50
         ELSE IF (SIGN(1.,Q).NE.SIGN(1.,QOLD) .AND. QOLD.NE.0) THEN
            WRITE(*,*) "I=", I, "  P=",P,
     &          "  Q = ", QOLD, " --> Q = ", Q
            DSTEP = DSTEP / 10.
            BOLD = POLD
            IF (DSTEP .LT. 1E-6) GOTO 50
            GOTO 10
         ENDIF
      ENDDO
c
      DSTEP = DSTEP / 10.
      IF (DSTEP .LT. 1E-6) THEN
         WRITE(*,*) "Nullstelle nicht gefunden"
         IQUE=1
         GOTO 999
      ENDIF
      GOTO 10
c
 50   B=P
c
      XQ20 = X2**B - X0**B
      A = (Y2-Y0 ) / XQ20
      C = Y0 - X0**B * A
c
c
      IF (A.EQ.B .AND. A.EQ.C) THEN
         IQUE=1
         WRITE(*,*) " GL-System nicht loesbar"
      ELSE
         IQUE=0
      ENDIF
c
c     *** The End
 999  RETURN
      END
c
c
C**********************************************************************
      SUBROUTINE MEDIAN(F, NMES, XMED, IQUE)
C**********************************************************************
c
c     ******** Sortieren und Berechnen des Medians eines REAL-Feldes
c
c     ******** Version 1.0 - Januar 1997 - by Hag
c
c     ***
c     ***    F = reales Feld
c     *** NMES = Groesse des Feldes F
c     *** XMED = Median des Feldes F
c     *** IQUE = Kommentarvariable, z.Zt. Dummy
c     ***
c     ***
c
      REAL F(*), XMED
      IAB=0
c
c     *** Sortieren des Feldes FDUM
      CALL SORT(F, NMES, IAB)
c
c     *** Medianposition
      IF (FLOAT(INT(NMES/2)) .EQ. FLOAT(NMES)/2.) THEN
         XMED = (F(NMES/2) + F(NMES/2 + 1)) / 2.
      ELSE
         XMED = F( (NMES+1)/2 )
      ENDIF
c
c     *** The End
 999  RETURN
      END
c
C**********************************************************************
      SUBROUTINE ROOTMSE(X,Y,N, RMSE, NRMSE, CVRMSE)
C**********************************************************************
C
c     ***     X = Zeireihe 1
c     ***     Y = Zeireihe 2 - Beobachtung
c     ***     N = Anzahl der Werte pro Zeitreihe
c
c     ***     Calculation of root mean square error: RMSE = SQRT( SUM( (xi - yi)^2)/n )
c     ***     Normalized RMSE = RSME / (xmax-ymin), yi = obs
c     ***     Coefficient of variation of the RMSD: CV(RMSE) =  RMSE / Mean(y)
c
C     ******************* Vereinbarungsteil ****************************
C
      PARAMETER (XMISS=-9999.,ZEPS=1.E-6)
      INTEGER N, I, IQUE, LU, NVAL
      REAL X(*),Y(*)
      REAL RMSE, NRMSE, CVRMSE
      REAL YMEAN, YMAX, YMIN, YSIG, CHI2SUM, XDUM
      LOGICAL GAUS
c
      REAL, DIMENSION(:), ALLOCATABLE :: F1
      REAL, DIMENSION(:), ALLOCATABLE :: F2
C
      ALLOCATE(F1(N))
      ALLOCATE(F2(N))
c
      IQUE=0
      LU=0
      GAUS=.FALSE.
c
      NVAL=0
      CHI2SUM = 0.
      DO I=1, N
         IF (ABS(X(I)-XMISS).GT.ZEPS
     &        .AND.ABS(Y(I)-XMISS).GT.ZEPS) THEN
           NVAL = NVAL+1
           F1(NVAL) = X(I)
           F2(NVAL) = Y(I)
           XDUM = X(I) - Y(I)
           CHI2SUM = CHI2SUM + XDUM*XDUM
         ENDIF
      ENDDO
      IF (NVAL.NE.N) WRITE(*,*) "ROOTSME: Missing values: ", N-NVAL
      RMSE = SQRT(CHI2SUM/FLOAT(NVAL))
c
      CALL MITTEL(F2,NVAL, YMEAN,YSIG,YMIN,YMAX,LU,GAUS,IQUE)
      NRMSE = RMSE / (YMAX-YMIN)
      CVRMSE = RMSE / YMEAN
c
      DEALLOCATE(F1)
      DEALLOCATE(F2)
c
c     *** The End
 999  RETURN
      END
