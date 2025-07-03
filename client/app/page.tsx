"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChevronDown, TrendingUp, Users, Target, Loader2 } from "lucide-react"

interface PredictionResult {
  winner: string
  win_probability: number
}

/* const players = ["Aaron Krickstein","Adrian Mannarino","Adrian Ungur","Adrian Voinea","Adriano Panatta","Agustin Calleri","Aisam Ul Haq Qureshi","Aki Rahunen","Albert Costa","Albert Montanes","Albert Portas","Albert Ramos","Alberto Berasategui","Alberto Mancini","Alberto Martin","Alberto Tous","Alejandro Davidovich Fokina","Alejandro Falla","Alejandro Ganzabal","Alejandro Gonzalez","Alejandro Hernandez","Alejandro Olmedo","Alejandro Pierola","Alejandro Tabilo","Aleksandar Vukic","Alessio Di Mauro","Alex Antonitsch","Alex Bogomolov Jr","Alex Calatrava","Alex Corretja","Alex De Minaur","Alex Lopez Moron","Alex Metreveli","Alex Michelsen","Alex Molcan","Alex Obrien","Alex Radulescu","Alexander Bublik","Alexander Mronz","Alexander Peya","Alexander Popp","Alexander Shevchenko","Alexander Volkov","Alexander Waske","Alexander Zverev","Alexandr Dolgopolov","Alexandre Muller","Alexei Popyrin","Aljaz Bedene","Allan Stone","Alvaro Betancur","Alvaro Fillol","Alvin Gardiner","Amer Delic","Amos Mansdorf","Anand Amritraj","Anders Jarryd","Andre Agassi","Andre Sa","Andrea Gaudenzi","Andreas Beck","Andreas Haider Maurer","Andreas Maurer","Andreas Seppi","Andreas Vinciguerra","Andrei Cherkasov","Andrei Chesnokov","Andrei Medvedev","Andrei Olhovskiy","Andrei Pavel","Andrei Stoliarov","Andrej Martin","Andres Gimeno","Andres Gomez","Andrew Castle","Andrew Ilie","Andrew Jarrett","Andrew Pattison","Andrew Sznajder","Andrey Golubev","Andrey Kuznetsov","Andrey Rublev","Andy Andrews","Andy Kohlberg","Andy Murray","Andy Roddick","Angel Gimenez","Antonio Munoz","Antonio Zugarelli","Antony Dupuis","Aqeel Khan","Armistead Neely","Arnaud Boetsch","Arnaud Clement","Arnaud Di Pasquale","Arne Thoms","Arthur Ashe","Arthur Fils","Arthur Rinderknech","Aslan Karatsev","Attila Balazs","Attila Korpas","Attila Savolt","Balazs Taroczy","Barry Moir","Barry Phillips Moore","Bart Wuyts","Belus Prajoux","Ben Shelton","Ben Testerman","Benjamin Balleret","Benjamin Becker","Benjamin Bonzi","Benoit Paire","Bernabe Zapata Miralles","Bernard Boileau","Bernard Fritz","Bernard Mignot","Bernard Tomic","Bernd Karbacher","Bill Bowrey","Bill Lloyd","Bill Scanlon","Billy Martin","Birger Andersson","Bjorn Borg","Bjorn Fratangelo","Bjorn Phau","Blaine Willenborg","Blaz Kavcic","Bob Bryan","Bob Carmichael","Bob Giltinan","Bob Green","Bob Hewitt","Bob Lutz","Bobby Reynolds","Bohdan Ulihrach","Boris Becker","Boris Pashanski","Borna Coric","Boro Jovanovic","Botic Van De Zandschulp","Brad Drewett","Brad Gilbert","Brad Pearce","Bradley Klahn","Brandon Nakashima","Brett Steven","Brian Baker","Brian Fairlie","Brian Gottfried","Brian Macphie","Brian Teacher","Brian Vahaly","Broderick Dyke","Bruce Derlin","Bruce Manson","Bruno Oresar","Bryan Shelton","Bud Schultz","Buster C Mottram","Butch Seewagen","Butch Walts","Byron Bertram","Byron Black","Cameron Norrie","Carl Limberger","Carl Uwe Steeb","Carlos Alcaraz","Carlos Berlocq","Carlos Castellan","Carlos Costa","Carlos Kirmayr","Carlos Moya","Carsten Arriens","Casper Ruud","Cassio Motta","Cecil Mamiit","Cedric Pioline","Cedrik Marcel Stebe","Charlie Fancutt","Charlie Pasarell","Chip Hooper","Chris Delaney","Chris Dunk","Chris Garner","Chris Guccione","Chris Johnstone","Chris Kachel","Chris Lewis","Chris Mayotte","Chris Pridham","Chris Wilkinson","Chris Woodruff","Christian Bergstrom","Christian Kuhnke","Christian Miniussi","Christian Ruud","Christian Saceanu","Christian Vinck","Christo Steyn","Christo Van Rensburg","Christophe Casa","Christophe Freyss","Christophe Rochus","Christopher Eubanks","Christopher Oconnell","Chuck Adams","Clark Graebner","Claudio Mezzadri","Claudio Panatta","Claudio Pistolesi","Cliff Drysdale","Cliff Letcher","Cliff Richey","Colin Dibley","Colin Dowdeswell","Constant Lestienne","Corentin Moutet","Corrado Barazzutti","Craig A Miller","Craig Wittus","Cristian Garin","Cristiano Caratti","Cyril Saulnier","Dale Collings","Damir Dzumhur","Damir Keretic","Dan Cassidy","Dan Goldie","Danai Udomchoke","Danie Visser","Daniel Altmaier","Daniel Brands","Daniel Contet","Daniel Elahi Galan","Daniel Evans","Daniel Gimeno Traver","Daniel Koellerer","Daniel Nestor","Daniel Vacek","Daniele Bracciali","Daniil Medvedev","Danilo Marcelino","Danny Sapsford","Darian King","Darren Cahill","David Carter","David De Miguel","David Engel","David Ferrer","David Goffin","David Lloyd","David Mustard","David Nainkin","David Nalbandian","David Pate","David Prinosil","David Rikl","David Sanchez","David Schneider","David Wheaton","Davide Sanguinetti","Denis Gremelmayr","Denis Istomin","Denis Kudla","Denis Shapovalov","Dennis Novak","Dennis Ralston","Dennis Van Scheppingen","Deon Joubert","Derek Schroder","Derek Tarr","Derrick Rostagno","Di Wu","Dick Bohrnstedt","Dick Crealy","Dick Dell","Dick Norman","Dick Stockton","Diego Nargiso","Diego Perez","Diego Schwartzman","Dimitri Poliakov","Dinu Pescariu","Dmitry Tursunov","Dominic Thiem","Dominik Hrbaty","Dominik Koepfer","Dominique Bedel","Donald Young","Doug Crawford","Doug Flach","Douglas Palm","Drew Gitlin","Dudi Sela","Dusan Lajovic","Dustin Brown","Eddie Dibbs","Eddie Edwards","Edouard Roger-Vasselin","Eduardo Bengoechea","Eduardo Masso","Eduardo Schwank","Egor Gerasimov","Elias Ymer","Eliot Teltscher","Emil Ruusuvuori","Emilio Benfele Alvarez","Emilio Gomez","Emilio Montano","Emilio Sanchez","Eric Deblicker","Eric Friedler","Eric Fromm","Eric Jelen","Eric Korita","Eric Winogradsky","Erick Iskersky","Erik Van Dillen","Ernesto Escobedo","Ernests Gulbis","Ernie Ewert","Evgeny Donskoy","Evgeny Korolev","Eyal Ran","Ezio Di Matteo","Fabian Marozsan","Fabio Fognini","Fabrice Santoro","Facundo Bagnis","Federico Coria","Federico Delbonis","Feliciano Lopez","Felix Auger Aliassime","Felix Mantilla","Ferdi Taygan","Fernando Gonzalez","Fernando Luna","Fernando Meligeni","Fernando Roese","Fernando Verdasco","Fernando Vicente","Filip Dewulf","Filip Krajinovic","Filippo Volandri","Flavio Cipolla","Flavio Cobolli","Flavio Saretta","Florent Serra","Florian Mayer","Florin Segarceanu","Frances Tiafoe","Francesco Cancellotti","Francisco Cerundolo","Francisco Clavet","Francisco Gonzalez","Francisco Maciel","Francisco Montana","Francisco Roig","Francisco Yunis","Franco Davin","Franco Squillari","Francois Jauffret","Frank Dancevic","Frank Froehling","Frank Gebert","Frank Sedgman","Frantisek Pala","Fred Mcnair","Fred Stolle","Frederic Fontang","Frederico Gil","Frederik Fetterlein","Frederik Nielsen","Frew Mcmillan","Fritz Buehning","Gabriel Markus","Gabriel Urpi","Gael Monfils","Galo Blanco","Gary Muller","Gastao Elias","Gaston Etlis","Gaston Gaudio","Gene Mayer","Gene Scott","Geoff Masters","George Bastl","George Hardie","George Kalovelonis","Georges Goven","Gerald Battrick","Gerald Melzer","Gerard Solves","German Lopez","Gianluca Mager","Gianluca Pozzi","Gianni Ocleppo","Gilad Bloom","Gilbert Schaller","Gilles Moretton","Gilles Muller","Gilles Simon","Giovanni Lapentti","Givaldo Barbosa","Glenn Layendecker","Glenn Michibata","Go Soeda","Goran Ivanisevic","Goran Prpic","Gouichi Motomura","Graham Stilwell","Grant Connell","Grant Stafford","Greg Holmes","Greg Rusedski","Grega Zemlja","Gregoire Barrere","Gregory Carraz","Grigor Dimitrov","Grover Raz Reid","Guido Pella","Guillaume Raoux","Guillermo Canas","Guillermo Coria","Guillermo Garcia Lopez","Guillermo Perez Roldan","Guillermo Vilas","Gustavo Kuerten","Guy Forget","Hank Pfister","Hans Dieter Beutel","Hans Gildemeister","Hans Joachim Ploetz","Hans Jurgen Pohmann","Hans Kary","Hans Schwaier","Hans Simonsson","Harald Elschenbroich","Harel Levy","Harold Solomon","Haroon Ismail","Haroon Rahim","Heinz Gunthardt","Hendrik Dreekmann","Henri Laaksonen","Henri Leconte","Henrik Holm","Henrik Sundstrom","Henry Bunis","Herb Fitzgibbon","Hernan Gumy","Hicham Arazi","Holger Rune","Horacio De La Pena","Horacio Zeballos","Horst Skoff","Howard Schoenfield","Hubert Hurkacz","Hugo Dellien","Hugo Gaston","Humphrey Hose","Huub Van Boeckel","Hyeon Chung","Hyung Taik Lee","Ian Crookenden","Ian Fletcher","Igor Andreev","Igor Kunitsyn","Igor Sijsling","Ilie Nastase","Illya Marchenko","Ilya Ivashka","Ion Tiriac","Irakli Labadze","Ismail El Shafei","Ivan Dodig","Ivan Kley","Ivan Lendl","Ivan Ljubicic","Ivan Miranda","Ivan Molina","Ivan Navarro","Ivo Heuberger","Ivo Karlovic","Ivo Minar","J J Wolf","Jacco Eltingh","Jack Draper","Jack Sock","Jacobo Diaz","Jaidip Mukerjea","Jaime Fillol","Jaime Oncins","Jaime Pinto Bravo","Jaime Yzaga","Jairo Velasco","Jakob Hlasek","James Blake","James Chico Hagey","James Duckworth","James Ward","Jamie Morgan","Jan Apell","Jan Gunnarsson","Jan Hajek","Jan Hernych","Jan Kodes","Jan Kroslak","Jan Kukal","Jan Lennard Struff","Jan Michael Gambill","Jan Norback","Jan Siemerink","Jan Vacek","Janko Tipsarevic","Jannik Sinner","Jared Donaldson","Jared Palmer","Jarkko Nieminen","Jaroslav Navratil","Jasjit Singh","Jason Kubler","Jason Stoltenberg","Jaume Munar","Javier Frana","Javier Sanchez","Javier Soler","Jay Berger","Jay Lapidus","Jean Baptiste Chanfreau","Jean Claude Barclay","Jean Francois Caujolle","Jean Louis Haillet","Jean Loup Rouyer","Jean Philippe Fleurian","Jean Rene Lisnard","Jeff Austin","Jeff Borowiak","Jeff Morrison","Jeff Salzenstein","Jeff Simpson","Jeff Tarango","Jens Knippschild","Jenson Brooksby","Jeremy Bates","Jeremy Chardy","Jerome Golmard","Jerome Potier","Jerzy Janowicz","Jesse Levine","Jim Courier","Jim Delaney","Jim Grabb","Jim Gurfein","Jim Mcmanus","Jim Pugh","Jimmy Arias","Jimmy Brown","Jimmy Connors","Jimmy Wang","Jimy Szymanski","Jiri Granat","Jiri Hrebec","Jiri Lehecka","Jiri Novak","Jiri Vanek","Jiri Vesely","Jo-Wilfried Tsonga","Joachim Johansson","Joakim Nystrom","Joao Cunha Silva","Joao Soares","Joao Sousa","Joao Souza","Joaquin Loyo Mayo","Joern Renzenbrink","Joey Rive","Johan Anderson","Johan Carlsson","Johan Kriek","Johan Van Herck","John Alexander","John Andrews","John Austin","John Bartlett","John Clifton","John Cooper","John Feaver","John Fitzgerald","John Frawley","John Isner","John James","John Lloyd","John Marks","John McEnroe","John Millman","John Newcombe","John Paish","John Ross","John Sadri","John Van Lottum","John Whitlinger","John Yuill","Jonas Bjorkman","Jonas Svensson","Jonathan Canter","Jonathan Smith","Jonathan Stark","Jordan Thompson","Jordi Arrese","Jordi Burillo","Jorge Andrew","Jorge Lozano","Jose Acasuso","Jose Edison Mandarino","Jose Francisco Altur","Jose Higueras","Jose Lopez Maeso","Jose Luis Clerc","Jose Luis Damiani","Jose Rubin Statham","Jozef Kovalik","Juan Aguilera","Juan Albert Viloca Puig","Juan Antonio Marin","Juan Avendano","Juan Balcells","Juan Carlos Ferrero","Juan Gisbert","Juan Ignacio Chela","Juan Ignacio Londero","Juan Ignacio Muntanola","Juan Martin del Potro","Juan Monaco","Juan Pablo Varillas","Julian Alonso","Julian Ganzabal","Julien Benneteau","Julien Boutter","Julio Goes","Jun Kamiwazumi","Jun Kuki","Juncheng Shang","Jurgen Fassbender","Jurgen Melzer","Jurgen Zopp","Justin Gimelstob","Kamil Majchrzak","Karel Novacek","Karen Khachanov","Karim Alami","Karl Meiler","Karol Beck","Karol Kucera","Karsten Braasch","Kei Nishikori","Keith Richardson","Kelly Evernden","Kelly Jones","Ken Flach","Ken Rosewall","Kenichi Hirai","Kenneth Carlsen","Kenny De Schepper","Kenny Thorne","Kent Carlsson","Kevin Anderson","Kevin Curren","Kevin Kim","Kevin Ullyett","Kim Warwick","Kjell Johansson","Klaus Eberhard","Kristian Pless","Kristof Vliegen","Kyle Edmund","Larry Stefanki","Lars Anders Wahlgren","Lars Burgsmuller","Lars Jonsson","Laslo Djere","Laurence Tieleman","Lawson Duncan","Leander Paes","Leif Shiras","Leo Palin","Leonardo Lavalle","Leonardo Mayer","Liam Broady","Libor Pimek","Lionel Roux","Lito Alvarez","Lleyton Hewitt","Lloyd Bourne","Lloyd Harris","Loic Courteau","Lorenzo Musetti","Lorenzo Sonego","Louk Sanders","Luca Van Assche","Lucas Arnold Ker","Lucas Pouille","Luciano Darderi","Luis Adrian Morejon","Luis Herrera","Luis Horna","Luiz Mattar","Lukas Dlouhy","Lukas Lacko","Lukas Rosol","Lukasz Kubot","Luke Jensen","Mackenzie Mcdonald","Magnus Gustafsson","Magnus Larsson","Magnus Norman","Mal Anderson","Malek Jaziri","Malivai Washington","Mansour Bahrami","Manuel Orantes","Manuel Santana","Marat Safin","Marc Andrea Huesler","Marc Flur","Marc Gicquel","Marc Kevin Goellner","Marc Lopez","Marc Rosset","Marcel Freeman","Marcel Granollers","Marcello Lara","Marcelo Arevalo","Marcelo Filippini","Marcelo Ingaramo","Marcelo Rios","Marco Cecchinato","Marco Chiudinelli","Marcos Aurelio Gorriz","Marcos Baghdatis","Marcos Daniel","Marcos Giron","Marcos Hocevar","Marcos Ondruska","Mardy Fish","Marian Vajda","Mariano Puerta","Mariano Zabaleta","Marin Cilic","Marinko Matosevic","Mario Ancic","Mario Martinez","Marius Copil","Mark Cox","Mark Dickson","Mark Edmondson","Mark Farrell","Mark Knowles","Mark Koevermans","Mark Kratzmann","Mark Petchey","Mark Philippoussis","Mark Woodforde","Marko Ostoja","Markus Hantschk","Markus Hipfl","Markus Naewie","Markus Zoecke","Marsel Ilhan","Martin Jaite","Martin Klizan","Martin Laurendeau","Martin Lee","Martin Mulligan","Martin Rodriguez","Martin Sinner","Martin Strelba","Martin Vassallo Arguello","Martin Verkerk","Martin Wostenholme","Marton Fucsovics","Marty Davis","Marty Riessen","Marzio Martelli","Massimo Cierro","Mats Wilander","Matt Anger","Matt Doyle","Matt Mitchell","Matteo Arnaldi","Matteo Berrettini","Matthew Ebden","Matthias Bachinger","Maurice Ruah","Mauricio Hadad","Max Mirnyi","Max Purcell","Maxime Cressy","Maximilian Marterer","Maximo Gonzalez","Mel Purcell","Menno Oosting","Michael Berrer","Michael Chang","Michael Joyce","Michael Kohlmann","Michael Llodra","Michael Mmoh","Michael Robertson","Michael Russell","Michael Stich","Michael Tauson","Michael Tebbutt","Michael Westphal","Michal Przysiezny","Michal Tabara","Michel Kratochvil","Michiel Schapers","Mikael Pernfors","Mikael Tillstrom","Mikael Ymer","Mike Bauer","Mike Belkin","Mike Cahill","Mike De Palmer","Mike Estep","Mike Fishbach","Mike Leach","Mike Machette","Mikhail Kukushkin","Mikhail Youzhny","Milan Holecek","Milan Srejber","Milos Raonic","Miloslav Mecir","Miomir Kecmanovic","Mirza Basic","Mischa Zverev","Nduka Odizor","Neil Borwick","Neville Godwin","Nicholas Kalogeropoulos","Nick Kyrgios","Nick Saviano","Nicklas Kroon","Nicklas Kulti","Nicola Spear","Nicolas Almagro","Nicolas Devilder","Nicolas Escude","Nicolas Jarry","Nicolas Kiefer","Nicolas Lapentti","Nicolas Mahut","Nicolas Massu","Nicolas Pereira","Nikola Pilic","Nikolay Davydenko","Nikoloz Basilashvili","Noam Okun","Norbert Gombos","Norman Holmes","Novak Djokovic","Nuno Borges","Nuno Marques","Oleg Ogorodov","Oliver Gross","Oliver Marach","Olivier Delaitre","Olivier Mutis","Olivier Patience","Olivier Rochus","Olli Rahnasto","Omar Camporese","Onny Parun","Orlin Stanoytchev","Oscar Hernandez","Oscar Otte","Ove Bengtson","Owen Davidson","Pablo Andujar","Pablo Arraya","Pablo Carreno Busta","Pablo Cuevas","Paolo Bertolucci","Paolo Cane","Paolo Lorenzi","Paradorn Srichaphan","Pascal Portes","Pat Cash","Pat Cramer","Pat Dupre","Patrice Dominguez","Patricio Cornejo","Patricio Rodriguez Chi","Patrick Baur","Patrick Hombergen","Patrick McEnroe","Patrick Proisy","Patrick Rafter","Patrik Kuhnen","Paul Annacone","Paul Capdeville","Paul Chamberlin","Paul Gerken","Paul Goldstein","Paul Haarhuis","Paul Henri Mathieu","Paul Kronk","Paul Mcnamee","Paul Vojtischek","Paul Wekesa","Pavel Hutka","Pavel Kotov","Pavel Slozil","Pedro Cachin","Pedro Martinez","Pedro Rebolledo","Per Hjertquist","Pere Riba","Pete Sampras","Peter Doerner","Peter Doohan","Peter Elter","Peter Feigl","Peter Fleming","Peter Gojowczyk","Peter Luczak","Peter Lundgren","Peter Mcnamara","Peter Pokorny","Peter Rennert","Peter Szoke","Peter Wessels","Petr Korda","Phil Dent","Philipp Kohlschreiber","Philipp Petzschner","Piero Toci","Pierre Barthes","Pierre Hugues Herbert","Pieter Aldrich","Potito Starace","Prakash Amritraj","Premjit Lall","Quentin Halys","Radek Stepanek","Radomir Vasek","Radu Albot","Raemon Sluiter","Rafael Nadal","Rainer Schuettler","Rajeev Ram","Ramesh Krishnan","Ramiro Benavides","Ramkumar Ramanathan","Ramon Delgado","Raul Antonio Viver","Raul Ramirez","Rauty Krog","Ray Keldie","Ray Ruffels","Raymond Moore","Razvan Sabau","Reilly Opelka","Renzo Furlan","Renzo Olivo","Ricardas Berankis","Ricardo Acuna","Ricardo Cano","Ricardo Mello","Ricardo Ycaza","Richard Fromberg","Richard Gasquet","Richard Gonzalez","Richard Krajicek","Richard Lewis","Richard Matuszewski","Richard Meyer","Richard Russell","Richey Reneberg","Rick Fagel","Rick Fisher","Rick Leach","Ricki Osterthun","Rik De Voest","Rinky Hijikata","Robbie Weiss","Robby Ginepri","Robert Kendrick","Robert Machan","Robert Maud","Robert Seguso","Robert Vant Hof","Roberto Arguello","Roberto Azar","Roberto Bautista Agut","Roberto Carballes Baena","Roberto Carretero","Roberto Vizcaino","Robin Drysdale","Robin Haase","Robin Soderling","Robin Vik","Rod Frawley","Rod Laver","Rodney Harmon","Rodolphe Gilbert","Roger Dowdeswell","Roger Federer","Roger Smith","Roger Taylor","Rogerio Dutra Silva","Roko Karanusic","Roland Stadler","Rolf Gehring","Rolf Norberg","Rolf Thung","Roman Safiullin","Ronald Agenor","Roscoe Tanner","Ross Case","Roy Barth","Roy Emerson","Ruben Bemelmans","Ruben Ramirez Hidalgo","Rui Machado","Russell Simpson","Ryan Harrison","Ryan Sweeting","Salvatore Caruso","Sam Groth","Sam Querrey","Sammy Giammalva Jr","Sandon Stolle","Sandor Noszaly","Sandy Mayer","Santiago Giraldo","Santiago Ventura","Sargis Sargsian","Sashi Menon","Scott Davis","Scott Draper","Scott Mccain","Sean Sorensen","Sebastian Baez","Sebastian Korda","Sebastian Ofner","Sebastien Grosjean","Sebastien Lareau","Sergi Bruguera","Sergio Casal","Sergio Roitman","Sergiy Stakhovsky","Shahar Perkiss","Sherwood Stewart","Shlomo Glickstein","Shuzo Matsuoka","Simon Greul","Simon Youl","Simone Bolelli","Simone Colombo","Sjeng Schalken","Slava Dosedel","Slobodan Zivojinovic","Somdev Devvarman","Soon Woo Kwon","Stan Smith","Stan Wawrinka","Stanislav Birner","Stefan Edberg","Stefan Eriksson","Stefan Koubek","Stefan Simonsson","Stefano Galvani","Stefano Pescosolido","Stefano Travaglia","Stefanos Tsitsipas","Stephane Huet","Stephane Robert","Stephane Simian","Stephen Warboys","Steve Bryan","Steve Campbell","Steve Darcis","Steve Denton","Steve Docherty","Steve Faulk","Steve Guy","Steve Johnson","Steve Krulevitz","Steve Meister","Steve Shaw","Steve Turner","Suwandi Suwandi","Syd Ball","Szabolcz Baranyi","Tadeusz Nowicki","Takao Suzuki","Tallon Griekspoor","Tarik Benhabiles","Taro Daniel","Tatsuma Ito","Taylor Dent","Taylor Fritz","Teimuraz Kakulia","Tenny Svensson","Tennys Sandgren","Terry Addison","Terry Moor","Terry Rocavert","Terry Ryan","Teymuraz Gabashvili","Thanasi Kokkinakis","Thiago Monteiro","Thiago Seyboth Wild","Thiemo De Bakker","Thierry Ascione","Thierry Champion","Thierry Guardiola","Thierry Tulasne","Thomas Enqvist","Thomas Fabbiano","Thomas Hogstedt","Thomas Johansson","Thomas Muster","Thomaz Bellucci","Thomaz Koch","Tim Gullikson","Tim Henman","Tim Mayotte","Tim Smyczek","Tim Wilkison","Tito Vazquez","Tobias Kamke","Todd Martin","Todd Nelson","Todd Witsken","Todd Woodbridge","Tom Cain","Tom Edlefsen","Tom Gorman","Tom Gullikson","Tom Leonard","Tom Nijssen","Tom Okker","Toma Ovici","Tomas Behrend","Tomas Berdych","Tomas Carbonell","Tomas Machac","Tomas Martin Etcheverry","Tomas Nydahl","Tomas Smid","Tomas Zib","Tommy Haas","Tommy Ho","Tommy Paul","Tommy Robredo","Tony Giammalva","Tony Graham","Tony Roche","Torben Ulrich","Tore Meinecke","Toshiro Sakai","Trevor Allan","Trey Waltke","Tsuyoshi Fukui","Tuomas Ketola","Udo Riglewski","Ugo Humbert","Ulf Stenlund","Ulrich Marten","Ulrich Pinner","Van Winitsky","Vasek Pospisil","Veli Paloheimo","Victor Amaya","Victor Crivoi","Victor Estrella","Victor Hanescu","Victor Pecci","Vijay Amritraj","Viktor Troicki","Vincent Spadea","Vincent Van Patten","Vincenzo Franchitti","Vitas Gerulaitis","Vladimir Korotkov","Vladimir Voltchkov","Vladimir Zednik","Wally Masur","Wanaro N'Godrella","Warren Maher","Wayne Arthurs","Wayne Black","Wayne Ferreira","Wayne Odesnik","Werner Eschauer","Werner Zirngibl","Wesley Moodie","William Brown","Wojtek Fibak","Wolfgang Popp","Xavier Malisse","Yahiya Doumbia","Yannick Hanfmann","Yannick Noah","Yen Hsun Lu","Yevgeny Kafelnikov","Yoshihito Nishioka","Younes El Aynaoui","Yuichi Sugita","Yuki Bhambri","Zan Guerry","Ze Zhang","Zeljko Franulovic","Zhizhen Zhang","Zizou Bergs","Zoltan Kuharszky"]
*/ 

export default function TennisPredictionApp() {
  const [player1, setPlayer1] = useState("")
  const [player2, setPlayer2] = useState("")
  const [surface, setSurface] = useState("")
  const [scrollLocked, setScrollLocked] = useState(true)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [predictionError, setPredictionError] = useState("")
  const [players, setPlayers] = useState<string[]>([])
  const subpageRef = useRef<HTMLDivElement>(null);

  // Lock scroll on component mount
  useEffect(() => {
    if (scrollLocked) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }

    // Cleanup function to restore scroll when component unmounts
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [scrollLocked])

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const response = await fetch("/api/players");
        if (!response.ok) throw new Error("Failed to fetch players");
        const data = await response.json();
        setPlayers(data.players || []);
      } catch (err) {
        console.log(err)
        throw new Error('Could not load players.');
      }
    };
    fetchPlayers();
  }, []);

  const scrollToInput = () => {
    setScrollLocked(false); // Unlock scroll to allow smooth scroll
    const inputSection = document.getElementById("input-section");
    if (inputSection) {
      inputSection.scrollIntoView({ behavior: "smooth" });
      // After the scroll animation (e.g., 800ms), lock scroll again
      setTimeout(() => {
        setScrollLocked(true);
      }, 800); // Adjust this duration to match your scroll speed
    }
  }

  const handlePredict = async () => {
    if (!player1 || !player2 || !surface) return
    
    try {
      setPredictionLoading(true)
      setPredictionError("")
      setPredictionResult(null)
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          p1: player1,
          p2: player2,
          surface: surface
        }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to make prediction')
      }
      
      const data = await response.json()
      console.log("Response from model:", data)
      if (data.error) {
        throw new Error(data.error)
      }
      
      setPredictionResult(data)
      setTimeout(() => {
        subpageRef.current?.scrollIntoView({ behavior: "smooth" });
        setTimeout(() => {
          setScrollLocked(true);
        }, 800); // lock scroll after scroll animation
      }, 100);
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : 'Failed to make prediction')
    } finally {
      setPredictionLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#3a4a3a]">
      {/* Hero Section */}
      <section className="min-h-screen flex flex-col items-center justify-center px-4 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="bg-[#e8dcc0] rounded-xl p-8 md:p-12 shadow-2xl">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-[#4a5a4a] rounded-2xl p-6 text-white">
                <TrendingUp className="w-8 h-8 mb-4 text-[#e8dcc0]" />
                <h3 className="text-lg font-semibold mb-2">Advanced Analytics</h3>
                <p className="text-sm opacity-90">
                  Leveraging machine learning to analyze player statistics and match history
                </p>
              </div>
              <div className="bg-[#b8956a] rounded-2xl p-6 text-white">
                <Users className="w-8 h-8 mb-4 text-white" />
                <h3 className="text-lg font-semibold mb-2">Player Insights</h3>
                <p className="text-sm opacity-90">
                  Deep analysis of playing styles, strengths, and performance patterns
                </p>
              </div>
              <div className="bg-[#4a5a4a] rounded-2xl p-6 text-white">
                <Target className="w-8 h-8 mb-4 text-[#e8dcc0]" />
                <h3 className="text-lg font-semibold mb-2">Match Prediction</h3>
                <p className="text-sm opacity-90">
                  Accurate predictions based on surface type and head-to-head records
                </p>
              </div>
            </div>
            <div className="text-center">
              <h2 className="text-3xl md:text-4xl font-bold text-[#3a4a3a] mb-4">Predict Tennis Match Outcomes</h2>
              <p className="text-lg text-[#3a4a3a] opacity-80 mb-8 max-w-2xl mx-auto">
                Our advanced machine learning model analyzes player statistics, playing styles, and surface preferences
                to provide accurate match predictions.
              </p>
              <Button
                onClick={scrollToInput}
                className="bg-[#3a4a3a] hover:bg-[#2a3a2a] text-white px-8 py-6 text-lg rounded-full transition-all duration-300 hover:scale-105"
              >
                Start Prediction
                <ChevronDown className="ml-2 w-5 h-5 animate-bounce" />
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Input Section */}
      <section id="input-section" className="min-h-screen py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">Match Setup</h2>
            <p className="text-xl text-white opacity-80">Enter player details and match conditions</p>
          </div>
          <div className="bg-[#e8dcc0] rounded-xl p-6 md:p-8 shadow-2xl">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Player 1 Section */}
              <Card className="bg-[#4a5a4a] border-none shadow-lg">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl text-white">Player 1</CardTitle>
                  <CardDescription className="text-[#e8dcc0]">Select first player</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="player1-select" className="text-white text-sm font-medium">
                      Player Name
                    </Label>
                    <Select value={player1} onValueChange={setPlayer1}>
                      <SelectTrigger className="mt-2 bg-white/10 border-white/20 text-white">
                        <SelectValue className='placeholder:text-white' placeholder="Choose player 1" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-black max-h-60">
                        {players.map((player) => (
                          <SelectItem key={player} value={player} className="hover:bg-gray-100">
                            {player}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                </CardContent>
              </Card>

              {/* Surface Type Section */}
              <Card className="bg-[#b8956a] border-none shadow-lg">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl text-white">Surface</CardTitle>
                  <CardDescription className="text-white/80">Select playing surface</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <Label htmlFor="surface-type" className="text-white text-sm font-medium">
                      Court Surface
                    </Label>
                    <Select value={surface} onValueChange={setSurface}>
                      <SelectTrigger className="mt-2 bg-white/10 border-white/20 text-white">
                        <SelectValue className='text-white' placeholder="Choose surface" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-black">
                        <SelectItem value="Hard" className="hover:bg-gray-100">Hard</SelectItem>
                        <SelectItem value="Clay" className="hover:bg-gray-100">Clay</SelectItem>
                        <SelectItem value="Grass" className="hover:bg-gray-100">Grass</SelectItem>
                        <SelectItem value="Carpet" className="hover:bg-gray-100">Carpet</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="pt-4">
                    <Button
                      onClick={handlePredict}
                      className="w-full bg-[#3a4a3a] hover:bg-[#2a3a2a] text-white py-3 text-lg rounded-xl transition-all duration-300 hover:scale-105"
                      disabled={!player1 || !player2 || !surface || predictionLoading}
                    >
                      {predictionLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Predicting...
                        </>
                      ) : (
                        'Predict Match'
                      )}
                    </Button>
                  </div>
                  
                  {/* Prediction Results */}
                  {predictionError && (
                    <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                      <p className="font-medium">Prediction Error:</p>
                      <p>{predictionError}</p>
                    </div>
                  )}
                  
                  {predictionResult && (
                    <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg">
                      <h4 className="font-bold text-lg mb-2">Prediction Result</h4>
                      <div className="space-y-2">
                        <p><strong>Winner:</strong> {predictionResult.winner}</p>
                        <p><strong>Confidence:</strong> {(predictionResult.win_probability).toFixed(2)}%</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Player 2 Section */}
              <Card className="bg-[#4a5a4a] border-none shadow-lg">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl text-white">Player 2</CardTitle>
                  <CardDescription className="text-[#e8dcc0]">Select second player</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="player2-select" className="text-white text-sm font-medium">
                      Player Name
                    </Label>
                    <Select value={player2} onValueChange={setPlayer2}>
                      <SelectTrigger className="mt-2 bg-white/10 border-white/20 text-white">
                        <SelectValue className='text-white' placeholder="Choose player 2" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-black max-h-60">
                        {players.map((player) => (
                          <SelectItem
                            key={player}
                            value={player}
                            className="hover:bg-gray-100"
                            disabled={player === player1}
                          >
                            {player}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}