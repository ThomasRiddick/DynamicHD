      PROGRAM HDFILE
c
c
c     ******* Program, dass die Inputfiles HD-Modells in einen Serviceformat-file
c             schreibt.
c
c     ******** Version 1.0 - Oktober 1999
c              Programmierung und Entwicklung: Stefan Hagemann
c
c     ******** Version 1.1 - Oktober 2003
c              IFORM = 2 - Version fuer XXF - SUN
c              Ohne Reservoirdateien - Switch IRES auf AUS=0
c
c
c     ******** Version 1.2 - Februar 2015
c              Possibility to add RDF-Index fields to HD parameter file
c              FILNEW for Longitude index, FIBNEW for latitude index of
c              destination grid box according to RDF FDIR.
c
c     ******* Linkroutine: GLOBUSE.f
c             Inputfile: global.inp
c
c
c     ********* Variablenliste
c     ***
c     ***     LU = allgemeine Logical Unit = 20
c     ***  LUINP = Logical Unit der binaeren Inputdatei = 50
c     ***  LUOUT = Logical Unit der binaeren Outputdatei = 60
c     ***
c     ***  IFORM = Datei-Format fuer globale felder fuer Routine GLREAD
c     ***          1 = Cray-Binaerformat fuer Cray-Lauf ohne Koord.-Trafo
c     ***          2 = REGEN: Globales Binaerformat
c     ***          3 = REGEN: Waveiso2-Format
c     ***          4 = Cray-Binaerformat mit Koord.-Trafo auf Datumsgrenze
c     ***
c     ******* Globale Arrays:
c     ***
c     *** FINMEM(NL, NB, NMEMRF) = Zwischenspeicherfeld der Speicherkaskade fuer
c     ***                        die  Inflows per Gridbox (max. 10 := NSMRF)
c     ***
c     *** FRUMEM(NL, NB, NMEMLF) = Zwischenspeicherfeld der Speicherkaskade fuer
c     ***                        die Runoffs per Gridbox (max. 5 := NSMLF)
c     ***
c     ***
c     ******* Sonstige Arrays
c     ***
c     ***  IHEAD = Header-Array fuer Craybinaer-Dateien
c     ***   AREA = Feld der Gitterboxflaechen, Einheit = [m^2]
c     ***   DELB = Konstanter latitudinaler Abstand, Einheit = [m]
c     *** DELTAL = Feld der longitudinalen Abstaende zwischen den
c     ***              Gitterboxmittelpunkten, Einheit = [m]
c     ***
c     ***
c     ******* Dateinamen
c     ***
c     *** DNGINP = global.inp = Inputdatei als Menueinitialisierung
c     ***
c     ******** Include of Parameter NL, NB
c     ***       NL, NB = Globale/Regionale Feldgrenzen
c     *** FLORG, FBORG = Ursprungskoordinaten
c     ***        FSCAL = Skalierungsfaktor = Breite einer Gitterbox in Grad
c
      INCLUDE 'pcom.for'
c
      PARAMETER (NMEMLF=5, NMEMRF=10)
      DIMENSION FDAT(NL, NB)
      REAL FDUM
      REAL  FINMEM(NL, NB, NMEMRF), FRUMEM(NL, NB, NMEMLF)
      REAL AREA(NB), DELTAL(NB), DELB
c
      DIMENSION FDIR(NL, NB)
      DIMENSION FIBNEW(NL,NB), FILNEW(NL,NB)
c
      INTEGER IHEAD(8)
      INTEGER LU, LUINP, LUOUT, IQUE, IFORM, IRES, IDUM
      INTEGER jl, il, jlnew, jb, ib, jbnew, idir

c
      CHARACTER*1024 DNINP, DNOUT
      CHARACTER*1024 DNGINP
      CHARACTER CINI*6, ZEILE*1024, DNAREA*1024, CF4*4
c
c     ********* Variablenbelegungen **********************************
c
      LU = 20
      LUINP = 50
      LUOUT = 60
      IFORM = 2
      IRES=0
      CF4=" "

      DNOUT="hdpara.srv"
      DNGINP = "global.inp"
c
c     *** Open Outputdatei
      OPEN(LUOUT, FILE=DNOUT, form='unformatted',
     &     IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNOUT
         WRITE(*,*) "******** Errornummer ",IOS
         GOTO 999
      ENDIF
c
      IHEAD(:)=0
      IHEAD(5)=NL
      IHEAD(6)=NB
c
c     ****** Inputdateien
c
c     *** 0. Land Sea Mask
      IHEAD(1)=172
      CINI = "TDNMAS"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
c
c     *** 1. River Direction File
      IHEAD(1)=701
      CINI = "TDNDIR"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDIR, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDIR
c
c     *** 2. Overlandflow - k-Parameter
      IHEAD(1)=702
      CINI = "TDNLFK"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
c
c     *** 3. Overlandflow - n-Parameter
      IHEAD(1)=703
      CINI = "TDNLFN"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
        XSUM=SUM(FDAT(:,:)/(NL*NB))
        WRITE(*,*) "OF N array: Mean = ", XSUM
        XMAX=MAXVAL(FDAT(:,:))
        XMIN=MINVAL(FDAT(:,:))
        WRITE(*,*) "      MAX = ", XMAX,"  MIN = ", XMIN
c
c     *** 4. Riverflow - k-Parameter
      IHEAD(1)=704
      CINI = "TDNRFK"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
c
c     *** 5. Riverflow - n-Parameter
      IHEAD(1)=705
      CINI = "TDNRFN"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
        XSUM=SUM(FDAT(:,:)/(NL*NB))
        WRITE(*,*) "RF N array: Mean = ", XSUM
        XMAX=MAXVAL(FDAT(:,:))
        XMIN=MINVAL(FDAT(:,:))
        WRITE(*,*) "      MAX = ", XMAX,"  MIN = ", XMIN
c
c     *** 6. Baseflow - k-Parameter
      IHEAD(1)=706
      CINI = "TDNGFK"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
c
c     *** Belegung des globalen Flaechenarrays: Nur Breitenabhaengig ==> NL=1
c     *** 7. Areas at 0.5 degree
      IHEAD(1)=707
      IHEAD(5)=1
      CINI = "TDNARE"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNAREA, IQUE)
      CALL AREAREAD(LU, DNAREA, AREA, DELTAL, DELB, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) AREA
      IHEAD(5)=NL
c
c     *** Index arrays?
      CINI = "IINDEX"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      IDUM = INT(FDUM+0.001)
      IF (IDUM.EQ.1) THEN
        FILNEW(:,:) = 0.
        FIBNEW(:,:) = 0.

        DO jb = 1, NB
        DO jl = 1, NL
          idir = INT(FDIR(jl, jb) + 0.001)
          IF (idir > 0) THEN          ! internal land
c
c           *** il, ib = relative direction coordinates [-1,0,1]
            ib = 1 - (idir - 1)/3
            il = MOD(idir - 1, 3) - 1

            FILNEW(jl,jb) = MOD(jl + il - 1 + nl, nl) + 1
            FIBNEW(jl,jb) = jb + ib

          ELSE                ! ocean and coast
            FILNEW(jl,jb) = jl
            FIBNEW(jl,jb) = jb
          END IF
        ENDDO
        ENDDO
c
        IHEAD(1)=708
        IHEAD(5)=NL
        WRITE(LUOUT) IHEAD
        WRITE(LUOUT) FILNEW
        IHEAD(1)=709
        WRITE(LUOUT) IHEAD
        WRITE(LUOUT) FIBNEW
      ENDIF
c
      CLOSE(LUOUT)
c
      IF (IRES.EQ.0) GOTO 999
c
c
c     ******** Auslesen der Kaskaden-Speicherinitialisierungsdateien
c     ******** Reduzieren auf benoetigte Reservoirs sowie zusammenpacken
c     ******** in den gleichen File
c
c     *** Open Outputdatei
      DNOUT="hdrstart.srv"
      OPEN(LUOUT, FILE=DNOUT, form='unformatted',
     &     IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNOUT
         WRITE(*,*) "******** Errornummer ",IOS
         GOTO 999
      ENDIF

c
c     *** 8. Overland flow and Riverflow reservoirs
      CINI = "TDNRES"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)

      OPEN(LUINP, FILE=DNINP,  form='unformatted',
     &      status='old', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateioeffnung ",DNINP
         WRITE(*,*) "******** Errornummer ",IOS
         GOTO 999
      ENDIF
      READ(LUINP) IHEAD
      READ(LUINP) FRUMEM
      WRITE(*,*) "Ini-Header Overlandflowkaskade: ", IHEAD
      DO I=1,NMEMLF
        XSUM=SUM(FRUMEM(:,:,I))/(NL*NB)
        WRITE(*,*) I,". OF Reservoir array: Mean = ", XSUM
        XMAX=MAXVAL(FRUMEM(:,:,I))
        XMIN=MINVAL(FRUMEM(:,:,I))
        WRITE(*,*) "      MAX = ", XMAX,"  MIN = ", XMIN
      ENDDO
c
c     *** Da standardmaessig N < 1.5 fuer Overland flow: ==> Linear  reservoir
      IHEAD(:)=0
      IHEAD(3)=999
      IHEAD(5)=NL
      IHEAD(6)=NB
      IHEAD(1)=710
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FRUMEM(:,:,1)
c
      READ(LUINP) IHEAD
      READ(LUINP) FINMEM
      WRITE(*,*) "Ini-Header Riverflowkaskade: ", IHEAD
      CLOSE(LUINP)
      DO I=1,NMEMRF
        XSUM=SUM(FINMEM(:,:,I))/(NL*NB)
        WRITE(*,*) I,". RF Reservoir array: Mean = ", XSUM
        XMAX=MAXVAL(FINMEM(:,:,I))
        XMIN=MINVAL(FINMEM(:,:,I))
        WRITE(*,*) "      MAX = ", XMAX,"  MIN = ", XMIN
      ENDDO
c
c     *** Da standard- N < 5.5 fuer River flow: ==> max. 5 reservoirs in cascade
c     *** Codes 711-715
      IHEAD(:)=0
      IHEAD(3)=999
      IHEAD(5)=NL
      IHEAD(6)=NB
      DO I=1, 5
         IHEAD(1)=710+I
         IHEAD(7) = I
         IHEAD(8) = 5
         WRITE(LUOUT) IHEAD
         WRITE(LUOUT) FINMEM(:,:,I)
      ENDDO
c
c     *** 9. Baseflow reservoir array
      IHEAD(1)=716
      CINI = "TDNGSP"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
c
c     *** 10. Inflow array --> For initialization
      IHEAD(1)=716
      CINI = "TDNINF"
      CALL GLOBINP(LU, DNGINP, CINI, FDUM, DNINP, IQUE)
      CALL GLREAD(LUINP, DNINP, IFORM, FDAT, NL, NB, IQUE)
      WRITE(LUOUT) IHEAD
      WRITE(LUOUT) FDAT
      CLOSE(LUOUT)
c
c     *** End of Program
 999  END
c

