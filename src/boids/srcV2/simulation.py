import numpy as np
from src.boids.srcV2.const import *

class Simulation:
    def __init__(self, num_boids: int):
        self.num_boids = num_boids
        
        # Initialisation des matrices globales (N, 2)
        self.positions = np.random.rand(num_boids, 2) * np.array(SIZE)
        self.angles = np.random.rand(num_boids) * 2 * np.pi
        
        self.velocities = np.empty((num_boids, 2))
        self.velocities[:, 0] = np.cos(self.angles) * MOVING_SPEED
        self.velocities[:, 1] = np.sin(self.angles) * MOVING_SPEED

        # Définition de la taille d'une case (basée sur le rayon d'attraction max)
        self.cell_size = ATTRACTION_RADIUS * CHARACTERISTIC_LENGTH

    def update(self):
        # --- STAGE 1 : CONSTRUCTION DE LA GRILLE (SPATIAL HASHING) ---
        # On détermine les coordonnées de la case (cx, cy) pour chaque boid d'un coup
        grid_coords = (self.positions // self.cell_size).astype(int)
        
        grid = {}
        for boid_idx, (cx, cy) in enumerate(grid_coords):
            cell_key = (cx, cy)
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(boid_idx)

        # Préparation des tableaux pour stocker les forces de cette frame
        repulsion_force = np.zeros((self.num_boids, 2))
        alignment_force = np.zeros((self.num_boids, 2))
        cohesion_force = np.zeros((self.num_boids, 2))
        
        # Tableau de booléens pour savoir qui subit de la répulsion
        has_repulsion = np.zeros(self.num_boids, dtype=bool)

        # Seuils de distance au carré (plus rapide à calculer, évite les racines carrées)
        r_rep_sq = (REPULSION_RADIUS * CHARACTERISTIC_LENGTH) ** 2
        r_ori_sq = (ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH) ** 2
        r_att_sq = (ATTRACTION_RADIUS * CHARACTERISTIC_LENGTH) ** 2

        # --- STAGE 2 : RECHERCHE DES VOISINS VIA LA GRILLE ---
        for i in range(self.num_boids):
            cx, cy = grid_coords[i]
            nearby_boids = []

            # On inspecte la case actuelle et les 8 cases adjacentes
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell_key = (cx + dx, cy + dy)
                    if cell_key in grid:
                        nearby_boids.extend(grid[cell_key])

            if len(nearby_boids) <= 1:
                continue  # Aucun voisin (seulement lui-même), on passe au boid suivant

            # Extraction des données des voisins potentiels
            nearby_indices = np.array(nearby_boids)
            nearby_indices = nearby_indices[nearby_indices != i] # On s'exclut de la liste

            # Calcul des distances uniquement avec ces voisins proches
            diff_pos = self.positions[nearby_indices] - self.positions[i]
            dist_sq = np.sum(diff_pos**2, axis=1)
            
            # --- NOUVEAU : FILTRE DE L'ANGLE MORT (FIELD OF VIEW) ---
            # 1. On calcule l'angle absolu de chaque voisin par rapport au boid i
            angles_to_neighbors = np.arctan2(diff_pos[:, 1], diff_pos[:, 0])
            
            # 2. On calcule la différence entre cet angle et l'angle de regard actuel du boid i
            angle_diffs = angles_to_neighbors - self.angles[i]
            
            # 3. On normalise cet écart entre -PI et PI
            angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
            
            # 4. Un voisin est visible uniquement si son écart angulaire est inférieur à la moitié du cône de vision
            in_view_mask = np.abs(angle_diffs) <= (VIEW_ANGLE / 2)
            
            # 5. On applique le filtre : on ne garde que les voisins visibles
            nearby_indices = nearby_indices[in_view_mask]
            diff_pos = diff_pos[in_view_mask]
            dist_sq = dist_sq[in_view_mask]

            if len(nearby_indices) == 0:
                continue # Si tous les voisins proches étaient dans l'angle mort, on passe au boid suivant

            # --- STAGE 3 : SÉPARATION DES FORCES PAR ZONE ---
            # Masques booléens locaux
            mask_rep = dist_sq <= r_rep_sq
            mask_ori = (dist_sq > r_rep_sq) & (dist_sq <= r_ori_sq)
            mask_att = (dist_sq > r_ori_sq) & (dist_sq <= r_att_sq)

            # 1. Répulsion (Séparation)
            if np.any(mask_rep):
                has_repulsion[i] = True
                rep_indices = nearby_indices[mask_rep]
                # Vecteur pointant à l'opposé du voisin
                diff_rep = self.positions[i] - self.positions[rep_indices]
                dists = np.sqrt(dist_sq[mask_rep])[:, np.newaxis]
                dists = np.where(dists < 1e-6, 1e-6, dists)
                
                # Plus ils sont proches, plus la force pousse fort
                repulsion_force[i] = np.sum((diff_rep / dists) / dists, axis=0)
                norm = np.linalg.norm(repulsion_force[i])
                if norm > 1e-6: repulsion_force[i] /= norm

            # 2. Alignement (Orientation)
            if np.any(mask_ori):
                ori_indices = nearby_indices[mask_ori]
                alignment_force[i] = np.sum(self.velocities[ori_indices], axis=0)
                norm = np.linalg.norm(alignment_force[i])
                if norm > 1e-6: alignment_force[i] /= norm

            # 3. Cohésion (Attraction)
            if np.any(mask_att):
                att_indices = nearby_indices[mask_att]
                center_of_mass = np.mean(self.positions[att_indices], axis=0)
                cohesion_force[i] = center_of_mass - self.positions[i]
                norm = np.linalg.norm(cohesion_force[i])
                if norm > 1e-6: cohesion_force[i] /= norm

        # --- STAGE 4 : ARBITRAGE VECTORIEL ET LOI DE VIRAGE ---
        has_repulsion_col = has_repulsion[:, np.newaxis]
        
        # Somme pondérée des comportements
        target_velocities = np.where(
            has_repulsion_col,
            self.velocities * 0.2 + repulsion_force * 0.8,
            self.velocities * 0.5 + alignment_force * 0.3 + cohesion_force * 0.2
        )

        # Limitation angulaire stricte (Gestion des petits virages)
        target_angles = np.arctan2(target_velocities[:, 1], target_velocities[:, 0])
        angle_diff = target_angles - self.angles
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        under_speed = np.abs(angle_diff) < TURNING_SPEED
        self.angles = np.where(under_speed, target_angles, self.angles + np.sign(angle_diff) * TURNING_SPEED)

        # --- STAGE 5 : PHYSIQUE, BRUIT ET BORDURES ---
        # Ajout du bruit
        self.angles += NOISE_ANGLE * (2 * np.random.rand(self.num_boids) - 1)

        # Actualisation du vecteur vitesse
        self.velocities[:, 0] = np.cos(self.angles) * MOVING_SPEED
        self.velocities[:, 1] = np.sin(self.angles) * MOVING_SPEED
        
        # Avancement
        self.positions += self.velocities

        # Traitement vectoriel des bordures de l'écran (Rebonds)
        left_mask = self.positions[:, 0] < 0
        self.positions[left_mask, 0] = 0
        self.velocities[left_mask, 0] *= -1

        right_mask = self.positions[:, 0] > SIZE[0]
        self.positions[right_mask, 0] = SIZE[0]
        self.velocities[right_mask, 0] *= -1

        top_mask = self.positions[:, 1] < 0
        self.positions[top_mask, 1] = 0
        self.velocities[top_mask, 1] *= -1

        bottom_mask = self.positions[:, 1] > SIZE[1]
        self.positions[bottom_mask, 1] = SIZE[1]
        self.velocities[bottom_mask, 1] *= -1

        # Synchronisation de l'angle pour les boids qui viennent de rebondir
        bounce_mask = left_mask | right_mask | top_mask | bottom_mask
        if np.any(bounce_mask):
            self.angles[bounce_mask] = np.arctan2(self.velocities[bounce_mask, 1], self.velocities[bounce_mask, 0])
