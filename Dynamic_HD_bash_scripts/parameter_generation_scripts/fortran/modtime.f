c****************************************************************************
      SUBROUTINE MODTIME(IMOD, TAU, XN, DX, AREA, SLOPE, DD,IQUE)
c****************************************************************************
c
c     ******* Routine welche die Lagtime nach verschiedenen
c             Modellen ausrechnet (empirische Formeln)
c
c     ******* Version 1.0 - Dez. 1996 - by Stefan Hagemann
c
c     *** Time of Concentration >= Lagtime
c     *** Soil Conservation Service:  ToC = 1.417 * Tlag
c
c
c        *** Special LagTime nach Taylor and Schwartz 1952 [Hour],
c        *** [AREA]=sq.miles, Lagtime = 1.176 * Tau2
c        *** DD = Drainage Density from Singh & also DD = Dauer of eff. Rainfall
c        *** ==> Drainage Density ist falsch!
ccc         W = 0.212 * (DX/1609.*DX/1609./2.)**(-0.36)
ccc         TAU = 0.6 / SQRT(SLOPE) * EXP(W*DD)
ccc         TAU = TAU *1.176 / 24.
c
c     ********* Variablenliste
c     ***
c     ***  IMOD = Modellart
c     ***   TAU = Lagtime between centroid of effective Rainfall and center of
c     ***         direct runoff [days]
c     ***         = time difference between centroids of input and output
c     ***    XN = Anzahl der linearen Speicher fuer Speicherkaskade
c     ***    DX = Streamlength [m]
c     ***  AREA = Watershed area [m^2]
c     *** SLOPE = DH/DX = Elevation Difference / DX
c     ***    DD = Drainage density, [%/km ]
c     ***
c
      PARAMETER(PI=3.1415927)
      PARAMETER (CSAU = 2., ALPHA = 0.1)
c
c     *** Umrechnung in 1/m und Minimum fuer Drainage Density festlegen
      IF (DD.LT.DX/AREA*100.*1000.) THEN
         IF (DD.NE.0) WRITE(*,*) "DD = ", DD, " --> ",
     &       DX/AREA *100.*1000. , " %/km"
         DD=DX/AREA
      ELSE
         DD = DD / 100. /1000.
      ENDIF
c
      IF (IMOD.EQ.1) THEN
c
c        *** Overlandflow - Sausen-Analogie nach Hagemann
         XN = 2.2214
         VSAU = CSAU * (SLOPE)**ALPHA
         TAU = 16.8522 * DX/171000. * 1.0885/VSAU * XN
      ELSE IF (IMOD.EQ.2) THEN
c
c        *** Riverflow - Sausen-Analogie nach Hagemann
         VSAU = CSAU * (SLOPE)**ALPHA
         XN = 9.1312
         TAU = 0.4112 * DX/228000. * 1.0039/VSAU * XN
      ELSE IF (IMOD.EQ.3) THEN
c
c        *** T.o.C. nach Plate: a1 gebietsabh., a3=-a2/2, entspricht Kirpich
         TAU = 0.06625 * (0.001*DX)**0.77 * SLOPE**(-0.385)
         TAU = TAU / 1.417 /24.
c
c        *** Time of Concentr.nach Kirpich [Hour] = 1/24 day, Tennessee
c        *** [DX]=feet, [slope] = dimensionless average slope
ccc         TAU = 0.00013 * (DX/0.3048)**0.77 * SLOPE**(-0.385)
ccc         TAU = TAU / 1.417 /24.
      ELSE IF (IMOD.EQ.4) THEN
c
c        *** Time of Concentr.nach Kirpich [Hour] = 1/24 day , Pennsylvania
c        *** [DX]=feet, [slope] = dimensionless average slope
         TAU = 21.67E-6 * (DX/0.3048)**0.77 * SLOPE**(-0.5)
         TAU = TAU / 1.417 /24.
      ELSE IF (IMOD.EQ.5) THEN
c
c        *** Time of Concentr.nach Johnstone & Cross [Hour],
c        *** [DX]=miles, [slope]=feet/mile
c        *** based on watersheds from 25 to 1624 sq.miles
         TAU = 5 * SQRT(DX/1609. /(SLOPE*1609/0.3048) )
         TAU = TAU / 1.417 /24.
      ELSE IF (IMOD.EQ.6) THEN
c
c        *** LagTime nach Johnstone & Cross 1949 [Hour],
c        *** [DX]=miles, [slope]=feet/mile
c        *** W = average basin width of watershed in miles
         W = SQRT(AREA)
         TAU = 1.5 + 90. * W / 1609. /(SLOPE*1609/0.3048)
         TAU = TAU / 24.
      ELSE IF (IMOD.EQ.7) THEN
c
c        *** Time of Concentr. f. Overlandflow nach Kerby [Min.]
c        *** [DX]=feet, [Slope]=DH/DX
c        *** N depends on surcace, siehe Tabelle (0.02 - 0.8)
         SN=0.8
         TAU = (2*(DX/0.3048)*SN/3./SQRT(SLOPE) )**(1./2.14)
         TAU = TAU / 1.417 /1440.
      ELSE IF (IMOD.EQ.8) THEN
c
c        *** Time of Concentr. Federal Aviation Agency, surface flow method
c        *** [DX] = feet = 30.48 cm
c        *** C = runoff coeff. of rational method
c        *** [SLOPE] % of longest overlandflow path
         C=0.4
         TAU = 0.03 * (1.1-C) * SQRT(DX/0.3048) * (SLOPE*100.)**0.333
         TAU = TAU / 1.417 /24.
      ELSE IF (IMOD.EQ.9) THEN
c
c        *** Lagtime: Nash-Modell (1962),
c        *** [AREA]=sq.miles = 258.998 ha (10000 m^2)
c        *** [DX]=miles = channel length = 1.609 km, [Tau] = hour
c        *** [slope]=parts per 10,000, d.h. in 1/10000 (mean basin slope)
         TAU = 27.6 * (AREA*0.0001/258.998)**0.3 *
     &         (SLOPE*10000.)**(-0.3)
         TAU = TAU/24.
         XN = 1. / (0.41 * (DX/1609.)**(-0.1))
      ELSE IF (IMOD.EQ.10) THEN
c
c        *** Lagtime: Nash-Modell (1962),
c        *** [DX]=miles = channel length = 1.609 km, [Tau] = hour
c        *** [slope]=feet/mile (mean slope of mainstream)
         TAU = 20. * (DX/1609.)**0.3 *
     &         (SLOPE*1609./0.3048)**(-0.33)
         TAU = TAU/24.
      ELSE IF (IMOD.EQ.11) THEN
c
c        *** Time of C. for watersheds areas < 50 sq.miles nach Williams [Hour]
c        *** [Area] = sq.miles, [D]= [DX] = Miles, [SLOPE] = average fall in
c        *** feet per 100 feet
         D = SQRT(AREA / PI) * 2.
         TAU = DX/D * (AREA*0.0001/258.998)**0.4 / ((SLOPE*100.)**0.2)
         TAU = TAU / 1.417 /24.
      ELSE IF (IMOD.EQ.12) THEN
c
c        *** LagTime nach Carter 1961 [Hour]
c        *** [DX]=miles, [SLOPE]=feet/mile, calculated by equal area method
         TAU = 1.7 * (DX/1609. / SQRT(SLOPE*1609./0.3048))**0.6
         TAU = TAU / 24.
      ELSE IF (IMOD.EQ.13) THEN
c
c        *** LagTime nach Wu 1963 [Hour]
c        *** [AREA]=sq.miles, [DX]=miles,
c        *** [slope]=feet/mile (mean slope of mainstream)
         TAU = 780.*(AREA*0.0001/258.998)**0.94 * (DX/1609.)**(-1.47)
     &         * (SLOPE*1609./0.3048)**(-1.47)
         TAU = TAU / 24.
      ELSE IF (IMOD.EQ.14) THEN
c
c        *** Kennedy & Watt [Hour]
c        *** [DX]=miles, B=1 + 20*(Lake+Wetland-Area)/AREA
c        *** [slope]=feet/mile (mean slope of mainstream)
         B=1 + 20*3/100.
         TAU = 6.71 * (DX/1609.)**0.66 * B**1.21
     &         * (SLOPE*1609./0.3048)**(-0.33)
         TAU = TAU / 24.
      ELSE IF (IMOD.EQ.15) THEN
c
c        *** o'Kelly Model, derived by Dooge, Time of Concentration [Hours]
c        *** [Area]=sq.miles,
c        *** [slope]=overland slope in parts per 10000.
         TAU = 2.58 * (AREA*0.0001/258.998)**0.41 /(SLOPE*10000.)**0.17
         TAU = TAU / 1.417 /24.
      ELSE IF (IMOD.EQ.16) THEN
c
c        *** o'Kelly Model, derived by Dooge, Retention Time k [Hours]
c        *** [Area]=sq.miles,        (N = TAU/k ???)
c        *** [slope]=overland slope in parts per 10000.
         TAU = 100.5 * (AREA*0.0001/258.998)**0.28 /(SLOPE*10000.)**0.7
         TAU = TAU /24.
      ELSE IF (IMOD.EQ.17) THEN
c
c        *** Special LagTime nach Hickok-Keppel-Rafferty 1959
c        *** [AREA]=acres, Lagtime = 1.176 * Tau2 [min]
c        *** DD = Drainage Density [feet/acre], [SLOPe] = %
c        *** 1 acre = 40.47 a (1 Ha = 100 a ==> 1 a = 100 m^2) = 4047 m^2
         TAU = 106. * ( (AREA/4047.)**0.3 /
     &         (SLOPE*100. * SQRT(DD*4047./0.3048)) )**0.61
         TAU = TAU *1.176 / 1440.
      ELSE IF (IMOD.EQ.18 .OR. IMOD.EQ.19) THEN
c
c        *** DVWK - Parallelspeicherkaskaden
c        *** DX = Vorfluterlaenge von Wasserscheide bis Kontrollpunkt [km]
c        *** SLOPE = Gefaelle ueber DX, [K1, K2] = Hours
         XK1 = 0.731 * (DX/1000./SQRT(SLOPE))**0.218
         XK2 = 3.04 * XK1**1.29
         XN=2.
         IF (IMOD.EQ.18) TAU = XK1*XN/24.
         IF (IMOD.EQ.19) TAU = XK2*XN/24.
      ENDIF
c
c        *** The End
      RETURN
      END
c


