# Q-learning
Projet Reinforcement Learning – CELLENZA 
 
Description de la situation 
 
➢ Lieu : Pièce d'une maison comportant trois dispositifs connectés : - Un ensemble de N lumières que l'on peut allumer ou éteindre ; - Un radiateur que l'on peut allumer ou éteindre ; - Un thermomètre indiquant la température ambiante de la pièce. 
 
➢ Sujets : Deux sujets sont à prendre en compte dans ce projet :  - Un bébé ; - Un parent. 
 
➢ Contraintes :  Plusieurs contraintes sont à prendre en compte : - Lorsque les lampes et/ou le chauffage est en marche, il y a une consommation électrique ce qui implique un coût plus ou moins élevé ; - Les ampoules des lampes n'éclairent pas toutes de la même façon (chaque ampoule en fonctionnement ne consomme pas la même quantité d'électricité) ; - Le bébé est achluophobe et souhaite évoluer dans un environnement relativement chaud ; - Le parent privilégie une économie au niveau de ses factures d'électricité quitte à vivre dans une pièce plus froide et moins éclairée. 
 
➢ Cas : Il y a quatre cas à prendre en compte : - Il n'y a ni le bébé ni le parent dans la pièce ; - Il n'y a que le bébé dans la pièce ; - Il n'y a que le parent dans la pièce ;  - Il y le bébé et le parent dans la pièce. 
 
 
Le but de ce projet est d’adapter de manière autonome l’atmosphère de la pièce (luminosité/température) en fonction des sujets qui s’y trouvent et en tenant compte des différentes contraintes. 
Pour répondre à ce problème on utilisera un algorithme de Reinforcement Learning où un agent prendra des actions (allumer une lampe, éteindre le chauffage, etc.) qui lui offriront ou non des récompenses en fonction du cas dans lequel on se trouve.  
Une caméra installée dans la pièce permettra de préciser dans quel cas on se trouve et transmettra l’information à l’agent qui prendra les actions adéquates en conséquence. 
