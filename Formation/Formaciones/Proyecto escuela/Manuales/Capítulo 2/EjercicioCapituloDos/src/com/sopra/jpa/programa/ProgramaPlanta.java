package com.sopra.jpa.programa;



import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;

import com.sopra.jpa.entities.Planta;
import com.sopra.jpa.servicio.ServicioPlanta;

public class ProgramaPlanta {

	public static void main(String[] args) {

		EntityManagerFactory emf = Persistence.createEntityManagerFactory("ProgramaPlanta");
		EntityManager em = emf.createEntityManager();
		ServicioPlanta servicio = new ServicioPlanta(em);

		// crear y persistir una planta
		em.getTransaction().begin();
		Planta p = servicio.crearPlanta(1,"Rosal","Arbusto" );
		em.getTransaction().commit();
		System.out.println("Planta " + p + " guardada");

		// buscar una planta
		p = servicio.buscarPlanta(1);
		System.out.println("Se ha encontrado la planta " + p);

		// buscar todas las plantas
		List<Planta> emps = servicio.buscarTodasLasPlantas ();
		for (Planta e : emps)
		System.out.println("Plantas encontradas: " + e);

		// modificar planta
		em.getTransaction().begin();
		p = servicio.cambiarTipo(1, "Floral");
		em.getTransaction().commit();
		System.out.println("Modificado " + p);

		// borrar una planta
		em.getTransaction().begin();
		servicio.borrarPlanta(1);
		em.getTransaction().commit();
		System.out.println("Planta " + p + " borrada");

		// cerrar el EM y EMF
		em.close();
		emf.close();
		}
}


