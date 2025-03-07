package com.sopra.jpa.servicio;

import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.TypedQuery;

import com.sopra.jpa.entities.Planta;

public class ServicioPlanta {
	
	protected EntityManager em;
	
	public ServicioPlanta(EntityManager em) {
		this.em= em;
	}
	
	public Planta crearPlanta( int id,String nombre, String tipo) {
		Planta p = new Planta();
		p.setId(id);
		p.setNombre(nombre);
		p.setTipo(tipo);
		em.persist(p);
		return p;
	}
	
	public void borrarPlanta(int id) {
		Planta p = buscarPlanta(id);
		if(p != null) {
			em.remove(p);
		}
	}
		
	public Planta buscarPlanta(int id) {
		return em.find(Planta.class, id);
	}
	
	public Planta cambiarTipo(int id, String nuevoTipo) {
		Planta p = buscarPlanta(id);
		if(p!= null) {
			p.setTipo(nuevoTipo);
		}
		return p;
		
	}
	
	public List<Planta> buscarTodasLasPlantas(){
		TypedQuery<Planta> query = em.createQuery("SELECT p FROM Planta p", Planta.class);
		return query.getResultList();
	}
	

}
